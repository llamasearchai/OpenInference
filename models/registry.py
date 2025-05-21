from typing import Dict, Any, List, Optional, Union
import os
import json
import time
import shutil
import logging
from dataclasses import dataclass, asdict, field
import uuid

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    name: str
    version: str
    framework: str
    task_type: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)

class ModelRegistry:
    """
    Registry for managing model versions and metadata.
    
    Provides functionality to register, retrieve, list, and delete models,
    as well as tracking model versions and metadata.
    """
    
    def __init__(self, registry_dir: str):
        """
        Initialize the model registry.
        
        Args:
            registry_dir: Directory to store model files and metadata
        """
        self.registry_dir = os.path.abspath(registry_dir)
        self.models_dir = os.path.join(self.registry_dir, "models")
        self.metadata_dir = os.path.join(self.registry_dir, "metadata")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Load existing metadata
        self.refresh()
    
    def refresh(self) -> None:
        """Reload metadata from disk."""
        self.models = {}
        
        # Load all metadata files
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.metadata_dir, filename), 'r') as f:
                        metadata = ModelMetadata.from_dict(json.load(f))
                        self.models[metadata.model_id] = metadata
                except Exception as e:
                    logger.error(f"Failed to load metadata from {filename}: {str(e)}")
    
    def register_model(self, 
                      name: str,
                      version: str,
                      framework: str,
                      task_type: str,
                      model_file: str,
                      description: str = "",
                      tags: List[str] = None,
                      metrics: Dict[str, Any] = None,
                      parameters: Dict[str, Any] = None,
                      additional_files: Dict[str, str] = None) -> ModelMetadata:
        """
        Register a new model in the registry.
        
        Args:
            name: Model name
            version: Model version
            framework: Framework used (pytorch, tensorflow, etc.)
            task_type: Type of task (classification, generation, etc.)
            model_file: Path to the main model file
            description: Model description
            tags: List of tags for the model
            metrics: Model performance metrics
            parameters: Model parameters and hyperparameters
            additional_files: Map of additional files to include (name -> path)
            
        Returns:
            metadata: Metadata for the registered model
        """
        # Generate a unique ID
        model_id = str(uuid.uuid4())
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model file
        model_filename = os.path.basename(model_file)
        model_dst_path = os.path.join(model_dir, model_filename)
        shutil.copy(model_file, model_dst_path)
        
        # Copy additional files
        files = {model_filename: model_dst_path}
        if additional_files:
            for file_name, file_path in additional_files.items():
                dst_path = os.path.join(model_dir, file_name)
                shutil.copy(file_path, dst_path)
                files[file_name] = dst_path
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            framework=framework,
            task_type=task_type,
            description=description,
            created_at=time.time(),
            tags=tags or [],
            metrics=metrics or {},
            parameters=parameters or {},
            files=files
        )
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, f"{model_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Add to models
        self.models[model_id] = metadata
        
        logger.info(f"Registered model {name} version {version} with ID {model_id}")
        return metadata
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)
    
    def get_model_path(self, model_id: str, file_name: Optional[str] = None) -> Optional[str]:
        """
        Get the path to a model file.
        
        Args:
            model_id: ID of the model
            file_name: Name of the file to retrieve (if None, returns the main model file)
            
        Returns:
            path: Path to the model file, or None if not found
        """
        metadata = self.get_model(model_id)
        if not metadata:
            return None
        
        if file_name is None:
            # Return the main model file (first one)
            if metadata.files:
                return next(iter(metadata.files.values()))
            return None
        
        # Return the specified file
        return metadata.files.get(file_name)
    
    def list_models(self, 
                   name: Optional[str] = None,
                   framework: Optional[str] = None,
                   task_type: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """
        List models matching the given criteria.
        
        Args:
            name: Filter by model name
            framework: Filter by framework
            task_type: Filter by task type
            tags: Filter by tags (all tags must match)
            
        Returns:
            models: List of matching model metadata
        """
        result = []
        
        for metadata in self.models.values():
            # Apply filters
            if name and metadata.name != name:
                continue
            if framework and metadata.framework != framework:
                continue
            if task_type and metadata.task_type != task_type:
                continue
            if tags and not all(tag in metadata.tags for tag in tags):
                continue
            
            result.append(metadata)
        
        # Sort by creation time (newest first)
        result.sort(key=lambda m: m.created_at, reverse=True)
        return result
    
    def update_model(self, model_id: str, **updates) -> Optional[ModelMetadata]:
        """
        Update model metadata.
        
        Args:
            model_id: ID of the model to update
            updates: Metadata fields to update
            
        Returns:
            metadata: Updated metadata, or None if model not found
        """
        metadata = self.get_model(model_id)
        if not metadata:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(metadata, key) and key != "model_id":
                setattr(metadata, key, value)
        
        # Save updated metadata
        metadata_path = os.path.join(self.metadata_dir, f"{model_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(f"Updated model {metadata.name} (ID: {model_id})")
        return metadata
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            success: Whether deletion was successful
        """
        metadata = self.get_model(model_id)
        if not metadata:
            return False
        
        # Delete model directory
        model_dir = os.path.join(self.models_dir, model_id)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        # Delete metadata file
        metadata_path = os.path.join(self.metadata_dir, f"{model_id}.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Remove from models
        del self.models[model_id]
        
        logger.info(f"Deleted model {metadata.name} (ID: {model_id})")
        return True