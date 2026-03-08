###############################################################################
# Variables — AI Research Intelligence Platform
###############################################################################

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "ai-research"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# ------------------ S3 ------------------
variable "s3_bucket_name" {
  description = "Name of the S3 bucket for document storage"
  type        = string
  default     = "ai-research-assistant-dev"
}

# ------------------ DynamoDB ------------------
variable "dynamodb_table_name" {
  description = "Name of the DynamoDB table for document metadata"
  type        = string
  default     = "ai-research-documents"
}

# ------------------ SQS ------------------
variable "sqs_queue_name" {
  description = "Name of the SQS queue for document processing"
  type        = string
  default     = "document-processing-queue"
}

variable "sqs_visibility_timeout" {
  description = "Visibility timeout for the SQS queue in seconds"
  type        = number
  default     = 300
}

variable "sqs_message_retention" {
  description = "Message retention period in seconds (4 days)"
  type        = number
  default     = 345600
}

variable "sqs_max_receive_count" {
  description = "Maximum receive count before sending to DLQ"
  type        = number
  default     = 3
}

# ------------------ Lambda ------------------
variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = "ai-research-document-processor"
}

variable "lambda_runtime" {
  description = "Lambda runtime"
  type        = string
  default     = "python3.11"
}

variable "lambda_memory_size" {
  description = "Lambda memory size in MB"
  type        = number
  default     = 512
}

variable "lambda_timeout" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 300
}

variable "lambda_handler" {
  description = "Lambda handler entrypoint"
  type        = string
  default     = "worker.lambda_handler"
}

# ------------------ IAM ------------------
variable "lambda_role_name" {
  description = "IAM role name for Lambda"
  type        = string
  default     = "ai-research-lambda-role"
}

variable "lambda_policy_name" {
  description = "IAM inline policy name for Lambda"
  type        = string
  default     = "ai-research-lambda-policy"
}

# ------------------ CloudWatch ------------------
variable "log_retention_days" {
  description = "CloudWatch log group retention in days"
  type        = number
  default     = 14
}

variable "lambda_error_threshold" {
  description = "Threshold for Lambda error alarm"
  type        = number
  default     = 0
}

variable "lambda_duration_threshold" {
  description = "Threshold for Lambda duration alarm in milliseconds (p95)"
  type        = number
  default     = 240000
}

# ------------------ Tags ------------------
variable "tags" {
  description = "Common tags applied to all resources"
  type        = map(string)
  default = {
    Project     = "ai-research-intelligence-platform"
    ManagedBy   = "terraform"
    Environment = "dev"
  }
}
