# API Gateway HTTP API — Setup Guide

This guide explains how to expose the FastAPI endpoints through an
**API Gateway HTTP API** (v2), which is simpler and cheaper than REST API.

---

## Prerequisites

- AWS CLI v2 configured with credentials
- FastAPI app running and accessible (Docker/EC2 with a public IP or load balancer)
- Note your FastAPI base URL, e.g. `http://<EC2-PUBLIC-IP>:8000`

---

## Endpoints to Expose

| Method   | Route                  | Description                     |
|----------|------------------------|---------------------------------|
| `POST`   | `/ingest`             | Upload and process a document   |
| `POST`   | `/query`              | Search the knowledge base       |
| `DELETE` | `/documents/{id}`     | Delete a document and vectors   |
| `GET`    | `/status/{id}`        | Get pipeline processing status  |
| `GET`    | `/health`             | Application health check        |

---

## Step 1: Create the HTTP API

```bash
aws apigatewayv2 create-api \
  --name ai-research-api \
  --protocol-type HTTP \
  --description "AI Research Intelligence Platform HTTP API" \
  --cors-configuration \
    AllowOrigins="*",AllowMethods="GET,POST,DELETE,OPTIONS",AllowHeaders="Content-Type,Authorization",MaxAge=86400
```

Save the `ApiId` from the output — you will need it in every subsequent command.

```bash
export API_ID=<your-api-id>
```

---

## Step 2: Create the Integration

The integration connects API Gateway to your FastAPI backend.

### Option A: Public URL Integration (simplest)

If your FastAPI app has a public URL (e.g. EC2 with a public IP):

```bash
aws apigatewayv2 create-integration \
  --api-id $API_ID \
  --integration-type HTTP_PROXY \
  --integration-method ANY \
  --integration-uri http://<EC2-PUBLIC-IP>:8000/{proxy} \
  --payload-format-version 1.0
```

### Option B: VPC Link Integration (private subnets)

If your FastAPI runs in a private subnet, create a VPC Link first:

```bash
# 1. Create VPC Link
aws apigatewayv2 create-vpc-link \
  --name ai-research-vpc-link \
  --subnet-ids subnet-xxx subnet-yyy \
  --security-group-ids sg-xxx

# Save the VpcLinkId
export VPC_LINK_ID=<your-vpc-link-id>

# 2. Create the integration using the VPC Link
aws apigatewayv2 create-integration \
  --api-id $API_ID \
  --integration-type HTTP_PROXY \
  --integration-method ANY \
  --connection-type VPC_LINK \
  --connection-id $VPC_LINK_ID \
  --integration-uri arn:aws:elasticloadbalancing:us-east-1:<ACCOUNT>:listener/app/<ALB-NAME>/<ALB-ID>/<LISTENER-ID> \
  --payload-format-version 1.0
```

Save the `IntegrationId` from the output:

```bash
export INTEGRATION_ID=<your-integration-id>
```

---

## Step 3: Create Routes

Create a route for each endpoint, all pointing to the same integration:

```bash
# POST /ingest
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "POST /ingest" \
  --target integrations/$INTEGRATION_ID

# POST /query
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "POST /query" \
  --target integrations/$INTEGRATION_ID

# DELETE /documents/{id}
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "DELETE /documents/{id}" \
  --target integrations/$INTEGRATION_ID

# GET /status/{id}
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "GET /status/{id}" \
  --target integrations/$INTEGRATION_ID

# GET /health
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "GET /health" \
  --target integrations/$INTEGRATION_ID
```

---

## Step 4: Create a Stage and Deploy

```bash
# Create the $default stage with auto-deploy enabled
aws apigatewayv2 create-stage \
  --api-id $API_ID \
  --stage-name '$default' \
  --auto-deploy
```

---

## Step 5: Get the Invoke URL

```bash
aws apigatewayv2 get-api --api-id $API_ID --query "ApiEndpoint" --output text
```

The invoke URL will look like:

```
https://<api-id>.execute-api.us-east-1.amazonaws.com
```

Test it:

```bash
curl https://<api-id>.execute-api.us-east-1.amazonaws.com/health
```

---

## CORS Configuration

CORS is configured at API creation time (Step 1) to match the existing FastAPI
CORS middleware settings:

| Setting          | Value                                 |
|------------------|---------------------------------------|
| Allow Origins    | `*`                                   |
| Allow Methods    | `GET, POST, DELETE, OPTIONS`          |
| Allow Headers    | `Content-Type, Authorization`         |
| Max Age          | `86400` (24 hours)                    |

To update CORS later:

```bash
aws apigatewayv2 update-api \
  --api-id $API_ID \
  --cors-configuration \
    AllowOrigins="*",AllowMethods="GET,POST,DELETE,OPTIONS",AllowHeaders="Content-Type,Authorization",MaxAge=86400
```

---

## Notes

- **HTTP API vs REST API**: HTTP API (v2) is recommended — it is simpler,
  faster, and up to 71% cheaper than REST API (v1).
- **Authentication**: Not configured here. For production, add a JWT
  authorizer or IAM authorization on the routes.
- **Custom domain**: Use `aws apigatewayv2 create-domain-name` to map a
  custom domain to the API.
