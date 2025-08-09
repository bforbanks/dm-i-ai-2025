import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Optional
import os
from pathlib import Path

# nnUNet v2 imports (required)
try:
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    from batchgenerators.utilities.file_and_folder_operations import load_json
    import nnunetv2
except ImportError as e:
    raise ImportError(f"nnUNet v2 is required for NNUNetV2Style model. Please install nnunetv2: {e}")

from models.base_model import BaseModel


class NNUNetV2Style(BaseModel):
    """
    NNUNet v2 style model for tumor segmentation.
    Integrates nnUNet v2 functionality while maintaining compatibility with the existing codebase.
    
    Key features:
    - Uses nnUNet v2 architecture and preprocessing
    - Maintains BaseModel integration for training/validation
    - Supports both training and inference modes
    - Compatible with existing data pipeline
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        lr=1e-3,
        weight_decay=1e-5,
        bce_loss_weight=0.5,
        false_negative_penalty=2.0,
        false_negative_penalty_scheduler: dict | None = None,
        patient_weight: float = 2.0,
        control_weight: float = 0.5,
        # nnUNet v2 specific parameters
        model_folder: str | None = None,
        configuration_name: str = "2d_fullres",
        use_folds: Union[Tuple[Union[int, str]], None] = None,
        checkpoint_name: str = 'checkpoint_final.pth',
        use_mirroring: bool = True,
        tile_step_size: float = 0.5,
    ):
        super().__init__(
            lr=lr, weight_decay=weight_decay, bce_loss_weight=bce_loss_weight, 
            false_negative_penalty=false_negative_penalty, false_negative_penalty_scheduler=false_negative_penalty_scheduler,
            patient_weight=patient_weight, control_weight=control_weight
        )
        
        # nnUNet v2 specific attributes
        self.model_folder = model_folder
        self.configuration_name = configuration_name
        self.use_folds = use_folds
        self.checkpoint_name = checkpoint_name
        self.use_mirroring = use_mirroring
        self.tile_step_size = tile_step_size
        
        # nnUNet v2 components
        self.plans_manager = None
        self.configuration_manager = None
        self.network = None
        self.dataset_json = None
        self.label_manager = None
        self.preprocessor = None
        
        # Initialize nnUNet v2 components if model folder is provided
        if self.model_folder and os.path.exists(self.model_folder):
            self._initialize_nnunetv2_components()
        else:
            raise ValueError("model_folder must be provided and must exist for NNUNetV2Style model")

    def _initialize_nnunetv2_components(self):
        """Initialize nnUNet v2 components from trained model folder"""
        try:
            # Load plans and configuration
            model_path = Path(self.model_folder) if self.model_folder else Path()
            plans = load_json(str(model_path / 'plans.json'))
            self.plans_manager = PlansManager(plans)
            self.configuration_manager = self.plans_manager.get_configuration(self.configuration_name)
            
            # Load dataset info
            self.dataset_json = load_json(str(model_path / 'dataset.json'))
            self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
            
            # Determine folds to use
            if self.use_folds is None:
                self.use_folds = nnUNetPredictor.auto_detect_available_folds(
                    self.model_folder, self.checkpoint_name
                )
            
            # Load network architecture
            num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            
            # Load trainer class and build network
            checkpoint = torch.load(
                str(Path(self.model_folder) / f'fold_{self.use_folds[0]}' / self.checkpoint_name),
                map_location=torch.device('cpu')
            )
            trainer_name = checkpoint['trainer_name']
            
            trainer_class = recursive_find_python_class(
                os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer"),
                trainer_name, 'nnunetv2.training.nnUNetTrainer'
            )
            
            if trainer_class is None:
                raise RuntimeError(f'Unable to locate trainer class {trainer_name}')
            
            self.network = trainer_class.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                self.label_manager.num_segmentation_heads,
                enable_deep_supervision=False
            )
            
            # Load weights
            self.network.load_state_dict(checkpoint['network_weights'])
            
            # Initialize preprocessor
            self.preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
            
            print(f"✅ Successfully initialized nnUNet v2 components from {self.model_folder}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize nnUNet v2 components: {e}")

    def forward(self, x):
        """Forward pass using nnUNet v2 network"""
        if self.network is None:
            raise RuntimeError("nnUNet v2 network not initialized. Please provide a valid model_folder.")
        
        # Ensure input is in the correct format for nnUNet v2
        if x.dim() == 4:  # [B, C, H, W]
            # nnUNet v2 expects [B, C, H, W] format
            output = self.network(x)
            # Apply sigmoid for binary segmentation
            return torch.sigmoid(output)
        else:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")

    def predict_with_nnunetv2(self, image_path: str, output_path: str | None = None) -> np.ndarray:
        """
        Predict using nnUNet v2 pipeline (requires initialized nnUNet v2 components)
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save output
            
        Returns:
            Segmentation mask as numpy array
        """
        if self.network is None or self.plans_manager is None:
            raise RuntimeError("nnUNet v2 components not initialized. Call with model_folder parameter.")
        
        # Load image using nnUNet v2 reader for PNG files
        rw = NaturalImage2DIO()
        image, props = rw.read_images([image_path])
        
        # Preprocess
        data, _ = self.preprocessor.run_case_npy(
            image, None, props, self.plans_manager, self.configuration_manager, self.dataset_json
        )
        
        # Convert to tensor
        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        
        # Predict
        with torch.no_grad():
            predicted_logits = self.network(data)
            predicted_probabilities = torch.sigmoid(predicted_logits)
            segmentation = (predicted_probabilities > 0.5).float()
        
        # Convert to numpy
        segmentation = segmentation.cpu().numpy()
        
        # Save if output path provided
        if output_path:
            # Use nnUNet v2's export function for proper saving
            rw.write_seg(segmentation, output_path, props)
        
        return segmentation

    def load_nnunetv2_model(self, model_folder: str):
        """Load nnUNet v2 model from folder"""
        self.model_folder = model_folder
        self._initialize_nnunetv2_components() 