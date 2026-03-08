###############################################################################
# Root Module — AI Research Intelligence Platform
###############################################################################

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }

  # Local backend — state stored on disk
  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "aws" {
  region = var.aws_region
}

# ─── Data Sources ────────────────────────────────────────────────────────────

data "aws_caller_identity" "current" {}

# ─── Modules ────────────────────────────────────────────────────────────────

# 1. S3 — Document Storage
module "s3" {
  source      = "./modules/s3"
  bucket_name = var.s3_bucket_name
  tags        = var.tags
}

# 2. DynamoDB — Document Metadata
module "dynamodb" {
  source     = "./modules/dynamodb"
  table_name = var.dynamodb_table_name
  tags       = var.tags
}

# 3. SQS — Processing Queue
module "sqs" {
  source             = "./modules/sqs"
  queue_name         = var.sqs_queue_name
  visibility_timeout = var.sqs_visibility_timeout
  message_retention  = var.sqs_message_retention
  max_receive_count  = var.sqs_max_receive_count
  tags               = var.tags
}

# 4. IAM — Lambda Execution Role
module "iam" {
  source               = "./modules/iam"
  role_name            = var.lambda_role_name
  policy_name          = var.lambda_policy_name
  s3_bucket_arn        = module.s3.bucket_arn
  dynamodb_table_arn   = module.dynamodb.table_arn
  sqs_queue_arn        = module.sqs.queue_arn
  sqs_dlq_arn          = module.sqs.dlq_arn
  aws_region           = var.aws_region
  account_id           = data.aws_caller_identity.current.account_id
  lambda_function_name = var.lambda_function_name
  tags                 = var.tags
}

# 5. Lambda — Document Processor + CloudWatch
module "lambda" {
  source                    = "./modules/lambda"
  function_name             = var.lambda_function_name
  runtime                   = var.lambda_runtime
  handler                   = var.lambda_handler
  memory_size               = var.lambda_memory_size
  timeout                   = var.lambda_timeout
  role_arn                  = module.iam.role_arn
  sqs_queue_arn             = module.sqs.queue_arn
  s3_bucket_name            = module.s3.bucket_name
  dynamodb_table_name       = module.dynamodb.table_name
  sqs_queue_name            = module.sqs.queue_name
  log_retention_days        = var.log_retention_days
  lambda_error_threshold    = var.lambda_error_threshold
  lambda_duration_threshold = var.lambda_duration_threshold
  tags                      = var.tags
}
