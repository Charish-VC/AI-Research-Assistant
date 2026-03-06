#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# deploy_lambda.sh — Package and deploy the AI Research document processor
#                    Lambda function to AWS.
#
# Usage:
#   chmod +x scripts/deploy_lambda.sh
#   ./scripts/deploy_lambda.sh
#
# Prerequisites:
#   - AWS CLI v2 configured with credentials
#   - IAM role "ai-research-lambda-role" already created
#     (see infrastructure/iam/lambda_role.json)
#   - Python 3.11 + pip available
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
FUNCTION_NAME="ai-research-document-processor"
RUNTIME="python3.11"
HANDLER="worker.lambda_handler"
MEMORY_SIZE=512
TIMEOUT=300
REGION="${AWS_REGION:-us-east-1}"
ROLE_NAME="ai-research-lambda-role"
ZIP_FILE="lambda_deployment.zip"

# Environment variables for the Lambda function
S3_BUCKET_NAME="${S3_BUCKET_NAME:-ai-research-assistant-dev}"
DYNAMODB_TABLE_NAME="${DYNAMODB_TABLE_NAME:-ai-research-documents}"
SQS_QUEUE_NAME="${SQS_QUEUE_NAME:-document-processing-queue}"

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAMBDA_SRC="$PROJECT_ROOT/src/lambda"
BUILD_DIR="$(mktemp -d)"

echo "============================================================"
echo " Lambda Deployment: $FUNCTION_NAME"
echo "============================================================"
echo ""
echo "Build directory: $BUILD_DIR"
echo ""

# ── Step 1: Install dependencies ─────────────────────────────────────────
echo "▸ Step 1/5: Installing dependencies..."
pip install \
  --target "$BUILD_DIR" \
  --requirement "$LAMBDA_SRC/requirements.txt" \
  --quiet \
  --no-cache-dir
echo "  ✔ Dependencies installed"

# ── Step 2: Copy Lambda handler ──────────────────────────────────────────
echo "▸ Step 2/5: Copying worker.py..."
cp "$LAMBDA_SRC/worker.py" "$BUILD_DIR/"
echo "  ✔ worker.py copied"

# ── Step 3: Create ZIP ───────────────────────────────────────────────────
echo "▸ Step 3/5: Creating deployment package..."
cd "$BUILD_DIR"
zip -r9 "$PROJECT_ROOT/$ZIP_FILE" . -x "*.pyc" "__pycache__/*" > /dev/null
cd "$PROJECT_ROOT"
ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "  ✔ Created $ZIP_FILE ($ZIP_SIZE)"

# ── Step 4: Get IAM role ARN ─────────────────────────────────────────────
echo "▸ Step 4/5: Resolving IAM role ARN..."
ROLE_ARN=$(aws iam get-role \
  --role-name "$ROLE_NAME" \
  --query "Role.Arn" \
  --output text \
  2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ]; then
  echo "  ✘ IAM role '$ROLE_NAME' not found."
  echo "    Create it first using the commands in infrastructure/iam/lambda_role.json"
  echo ""
  echo "    Quick setup:"
  echo "    1. Save the TrustPolicy from lambda_role.json to trust-policy.json"
  echo "    2. aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://trust-policy.json"
  echo "    3. Save the InlinePolicy.PolicyDocument to policy.json"
  echo "    4. aws iam put-role-policy --role-name $ROLE_NAME --policy-name ai-research-lambda-policy --policy-document file://policy.json"
  echo ""
  rm -rf "$BUILD_DIR" "$ZIP_FILE"
  exit 1
fi
echo "  ✔ Role ARN: $ROLE_ARN"

# ── Step 5: Create or update Lambda function ─────────────────────────────
echo "▸ Step 5/5: Deploying Lambda function..."

ENV_VARS="Variables={S3_BUCKET_NAME=$S3_BUCKET_NAME,DYNAMODB_TABLE_NAME=$DYNAMODB_TABLE_NAME,SQS_QUEUE_NAME=$SQS_QUEUE_NAME,AWS_REGION=$REGION}"

# Check if function already exists
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" > /dev/null 2>&1; then
  echo "  Function exists — updating..."
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_FILE" \
    --region "$REGION" \
    --no-cli-pager > /dev/null

  # Wait for the update to be processed
  echo "  Waiting for update to complete..."
  aws lambda wait function-updated \
    --function-name "$FUNCTION_NAME" \
    --region "$REGION" 2>/dev/null || sleep 5

  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --runtime "$RUNTIME" \
    --handler "$HANDLER" \
    --memory-size "$MEMORY_SIZE" \
    --timeout "$TIMEOUT" \
    --environment "$ENV_VARS" \
    --region "$REGION" \
    --no-cli-pager > /dev/null

  echo "  ✔ Lambda function updated"
else
  echo "  Function does not exist — creating..."
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --runtime "$RUNTIME" \
    --handler "$HANDLER" \
    --role "$ROLE_ARN" \
    --memory-size "$MEMORY_SIZE" \
    --timeout "$TIMEOUT" \
    --zip-file "fileb://$ZIP_FILE" \
    --environment "$ENV_VARS" \
    --region "$REGION" \
    --no-cli-pager > /dev/null

  echo "  ✔ Lambda function created"

  # Wait for function to become active
  echo "  Waiting for function to become active..."
  aws lambda wait function-active \
    --function-name "$FUNCTION_NAME" \
    --region "$REGION" 2>/dev/null || sleep 5
fi

# ── Attach SQS trigger ──────────────────────────────────────────────────
echo ""
echo "▸ Attaching SQS trigger..."

# Get the SQS queue ARN
QUEUE_URL=$(aws sqs get-queue-url \
  --queue-name "$SQS_QUEUE_NAME" \
  --region "$REGION" \
  --query "QueueUrl" \
  --output text)

QUEUE_ARN=$(aws sqs get-queue-attributes \
  --queue-url "$QUEUE_URL" \
  --attribute-names QueueArn \
  --region "$REGION" \
  --query "Attributes.QueueArn" \
  --output text)

# Check if event source mapping already exists
EXISTING_UUID=$(aws lambda list-event-source-mappings \
  --function-name "$FUNCTION_NAME" \
  --event-source-arn "$QUEUE_ARN" \
  --region "$REGION" \
  --query "EventSourceMappings[0].UUID" \
  --output text 2>/dev/null || echo "None")

if [ "$EXISTING_UUID" != "None" ] && [ -n "$EXISTING_UUID" ]; then
  echo "  SQS trigger already exists (UUID: $EXISTING_UUID)"
else
  aws lambda create-event-source-mapping \
    --function-name "$FUNCTION_NAME" \
    --event-source-arn "$QUEUE_ARN" \
    --batch-size 1 \
    --region "$REGION" \
    --no-cli-pager > /dev/null
  echo "  ✔ SQS trigger created (batch size: 1)"
fi

# ── Cleanup ──────────────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
echo ""
echo "============================================================"
echo " ✔ Deployment complete!"
echo ""
echo " Function: $FUNCTION_NAME"
echo " Runtime:  $RUNTIME"
echo " Memory:   ${MEMORY_SIZE}MB"
echo " Timeout:  ${TIMEOUT}s"
echo " Region:   $REGION"
echo " SQS:      $SQS_QUEUE_NAME → $FUNCTION_NAME"
echo "============================================================"
