###############################################################################
# Outputs — AI Research Intelligence Platform
###############################################################################

# ─── S3 ──────────────────────────────────────────────────────────────────────
output "s3_bucket_name" {
  description = "Name of the S3 document storage bucket"
  value       = module.s3.bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the S3 document storage bucket"
  value       = module.s3.bucket_arn
}

# ─── DynamoDB ────────────────────────────────────────────────────────────────
output "dynamodb_table_name" {
  description = "Name of the DynamoDB metadata table"
  value       = module.dynamodb.table_name
}

output "dynamodb_table_arn" {
  description = "ARN of the DynamoDB metadata table"
  value       = module.dynamodb.table_arn
}

# ─── SQS ─────────────────────────────────────────────────────────────────────
output "sqs_queue_url" {
  description = "URL of the document processing SQS queue"
  value       = module.sqs.queue_url
}

output "sqs_queue_arn" {
  description = "ARN of the document processing SQS queue"
  value       = module.sqs.queue_arn
}

output "sqs_dlq_url" {
  description = "URL of the dead-letter queue"
  value       = module.sqs.dlq_url
}

output "sqs_dlq_arn" {
  description = "ARN of the dead-letter queue"
  value       = module.sqs.dlq_arn
}

# ─── Lambda ──────────────────────────────────────────────────────────────────
output "lambda_function_name" {
  description = "Name of the Lambda document processor function"
  value       = module.lambda.function_name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda document processor function"
  value       = module.lambda.function_arn
}

# ─── IAM ─────────────────────────────────────────────────────────────────────
output "lambda_role_arn" {
  description = "ARN of the Lambda execution IAM role"
  value       = module.iam.role_arn
}

output "lambda_role_name" {
  description = "Name of the Lambda execution IAM role"
  value       = module.iam.role_name
}

# ─── CloudWatch ──────────────────────────────────────────────────────────────
output "cloudwatch_log_group_name" {
  description = "Name of the Lambda CloudWatch log group"
  value       = module.lambda.log_group_name
}
