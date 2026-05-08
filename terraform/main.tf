provider "aws" {
  region = "eu-north-1"
}

# Create a key pair for SSH access
resource "aws_key_pair" "mlops_key" {
  key_name   = "mlops-key"
  public_key = file("~/.ssh/id_ed25519.pub")
}

# Create security group with SSH access
resource "aws_security_group" "mlops_sg" {
  name        = "mlops-security-group"
  description = "Security group for MLops server"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "MLflow UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlops-security-group"
  }
}

resource "aws_s3_bucket" "seoul_bike_sharing_artifacts" {
  bucket = "seoul-bike-sharing-aphdinh-2"
  force_destroy = true
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "seoul_bike_sharing_server" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = "t3.micro"
  key_name      = aws_key_pair.mlops_key.key_name
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]

  tags = {
    Name = "mlops-server"
  }
}

# Outputs
output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.seoul_bike_sharing_server.id
}

output "instance_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.seoul_bike_sharing_server.public_ip
}

output "bucket_name" {
  description = "Name of the S3 bucket for artifacts"
  value       = aws_s3_bucket.seoul_bike_sharing_artifacts.bucket
}

output "region" {
  description = "AWS region"
  value       = "eu-north-1"
}