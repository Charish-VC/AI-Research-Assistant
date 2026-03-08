###############################################################################
# Lambda Module — Document Processor + CloudWatch Monitoring
###############################################################################

variable "function_name" {
  description = "Name of the Lambda function"
  type        = string
}

variable "runtime" {
  description = "Lambda runtime"
  type        = string
}

variable "handler" {
  description = "Lambda handler"
  type        = string
}

variable "memory_size" {
  description = "Lambda memory in MB"
  type        = number
}

variable "timeout" {
  description = "Lambda timeout in seconds"
  type        = number
}

variable "role_arn" {
  description = "ARN of the IAM execution role"
  type        = string
}

variable "sqs_queue_arn" {
  description = "ARN of the SQS queue to trigger Lambda"
  type        = string
}

variable "s3_bucket_name" {
  description = "S3 bucket name for environment variable"
  type        = string
}

variable "dynamodb_table_name" {
  description = "DynamoDB table name for environment variable"
  type        = string
}

variable "sqs_queue_name" {
  description = "SQS queue name for environment variable"
  type        = string
}

variable "log_retention_days" {
  description = "CloudWatch log group retention in days"
  type        = number
  default     = 14
}

variable "lambda_error_threshold" {
  description = "Threshold for Lambda error count alarm"
  type        = number
  default     = 0
}

variable "lambda_duration_threshold" {
  description = "Threshold for Lambda p95 duration alarm (ms)"
  type        = number
  default     = 240000
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {}
}

# ─── CloudWatch Log Group ────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.function_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

# ─── Lambda Function ────────────────────────────────────────────────────────

# A placeholder zip is required for initial import. After import, deploy your
# real code package via CI/CD or `aws lambda update-function-code`.
data "archive_file" "placeholder" {
  type        = "zip"
  output_path = "${path.module}/placeholder.zip"

  source {
    content  = "# placeholder — real code deployed separately"
    filename = "lambda_function.py"
  }
}

resource "aws_lambda_function" "processor" {
  function_name = var.function_name
  role          = var.role_arn
  handler       = var.handler
  runtime       = var.runtime
  memory_size   = var.memory_size
  timeout       = var.timeout

  filename         = data.archive_file.placeholder.output_path
  source_code_hash = data.archive_file.placeholder.output_base64sha256

  environment {
    variables = {
      S3_BUCKET_NAME      = var.s3_bucket_name
      DYNAMODB_TABLE_NAME = var.dynamodb_table_name
      SQS_QUEUE_NAME      = var.sqs_queue_name
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda]

  tags = var.tags

  # Ignore changes to code since deployments happen outside Terraform
  lifecycle {
    ignore_changes = [
      filename,
      source_code_hash,
    ]
  }
}

# ─── SQS Event Source Mapping ────────────────────────────────────────────────

resource "aws_lambda_event_source_mapping" "sqs" {
  event_source_arn = var.sqs_queue_arn
  function_name    = aws_lambda_function.processor.arn
  batch_size       = 1
  enabled          = true
}

# ─── CloudWatch Alarms ──────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.function_name}-errors"
  alarm_description   = "Triggers when Lambda error count exceeds threshold"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = var.lambda_error_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = aws_lambda_function.processor.function_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "lambda_duration" {
  alarm_name          = "${var.function_name}-duration-p95"
  alarm_description   = "Triggers when Lambda p95 duration exceeds threshold"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 300
  extended_statistic  = "p95"
  threshold           = var.lambda_duration_threshold
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = aws_lambda_function.processor.function_name
  }

  tags = var.tags
}

# ─── Outputs ─────────────────────────────────────────────────────────────────

output "function_name" {
  description = "Name of the Lambda function"
  value       = aws_lambda_function.processor.function_name
}

output "function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.processor.arn
}

output "log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.lambda.name
}

output "event_source_mapping_uuid" {
  description = "UUID of the SQS event source mapping"
  value       = aws_lambda_event_source_mapping.sqs.uuid
}
