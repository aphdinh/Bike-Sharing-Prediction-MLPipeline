import sys
import os
from typing import Dict, Any

def run_with_prefect() -> Dict[str, Any]:
    try:
        from .prefect_orchestrator import ml_training_pipeline
        print("Running with Prefect orchestration...")
        return ml_training_pipeline()
    except ImportError as e:
        print(f"Prefect not available: {e}")
        print("Falling back to core training...")
        return run_core_training()

def run_core_training() -> Dict[str, Any]:
    try:
        from .train_core import main_training_pipeline
        print("Running core training pipeline...")
        return main_training_pipeline()
    except ImportError as e:
        print(f"Core training module not available: {e}")
        raise

def create_prefect_deployment():
    try:
        from .prefect_orchestrator import create_deployment
        deployment = create_deployment()
        deployment.apply()
        print("Deployment created successfully!")
        print("Start an agent with: prefect agent start --pool default-agent-pool")
    except ImportError as e:
        print(f"Prefect not available: {e}")
        print("Cannot create deployment without Prefect.")

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "deploy":
            create_prefect_deployment()
        elif command == "prefect":
            result = run_with_prefect()
            print(f"Pipeline completed successfully!")
            print(f"Results: {result}")
        elif command == "core":
            result = run_core_training()
            print(f"Pipeline completed successfully!")
            print(f"Results: {result}")
        else:
            print(f"Unknown command: {command}")
    else:
        result = run_with_prefect()
        print(f"Pipeline completed successfully!")
        print(f"Results: {result}")


if __name__ == "__main__":
    main() 