###############################################################################
# DynamoDB Module — Document Metadata Table
###############################################################################

variable "table_name" {
  description = "Name of the DynamoDB table"
  type        = string
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {}
}

# ─── Resources ───────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "documents" {
  name         = var.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "doc_id"

  attribute {
    name = "doc_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = var.tags
}

# ─── Outputs ─────────────────────────────────────────────────────────────────

output "table_name" {
  description = "Name of the DynamoDB table"
  value       = aws_dynamodb_table.documents.name
}

output "table_arn" {
  description = "ARN of the DynamoDB table"
  value       = aws_dynamodb_table.documents.arn
}
