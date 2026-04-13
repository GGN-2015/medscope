import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPaintEvent
import os

# VTK imports
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VTKModelManager:
    """Manages 3D models in VTK scene."""
    
    def __init__(self, renderer: vtk.vtkRenderer):
        """
        Initialize the model manager.
        
        Args:
            renderer: VTK renderer to add models to
        """
        self.renderer: vtk.vtkRenderer = renderer
        self.models: Dict[str, vtk.vtkActor] = {}  # name -> actor
        self.model_sources: Dict[str, Any] = {}  # name -> source (for potential regeneration)
        
    def add_model_from_file(self, name: str, file_path: str, 
                           color: Optional[Tuple[float, float, float]] = None, 
                           scale: float = 1.0, 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        """
        Add a model from a file (supports .vtk, .stl, .obj, etc.)
        
        Args:
            name: Unique identifier for the model
            file_path: Path to the model file
            color: RGB tuple (0-1) or None for default
            scale: Scale factor
            position: (x, y, z) position
            
        Returns:
            bool: True if successful, False otherwise
        """
        if name in self.models:
            print(f"Model '{name}' already exists. Remove it first or use a different name.")
            return False
            
        # Determine reader based on file extension
        if file_path.lower().endswith('.stl'):
            reader = vtk.vtkSTLReader()
        elif file_path.lower().endswith('.obj'):
            reader = vtk.vtkOBJReader()
        elif file_path.lower().endswith('.vtk'):
            reader = vtk.vtkDataSetReader()
        elif file_path.lower().endswith('.ply'):
            reader = vtk.vtkPLYReader()
        else:
            print(f"Unsupported file format: {file_path}")
            return False
        
        if not os.path.isfile(file_path):
            print(f"Model file {file_path} not found.")
            return False

        reader.SetFileName(file_path)
        reader.Update()
        
        return self._add_model_from_algorithm(name, reader, color, scale, position)
    
    def _add_model_from_algorithm(self, name: str, algorithm: Any, 
                                  color: Optional[Tuple[float, float, float]],
                                  scale: float, 
                                  position: Tuple[float, float, float]) -> bool:
        """Internal method to add model from a VTK algorithm."""
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(algorithm.GetOutputPort())
        return self._add_model_from_mapper(name, mapper, color, scale, position)
    
    def _add_model_from_mapper(self, name: str, mapper: vtk.vtkPolyDataMapper,
                               color: Optional[Tuple[float, float, float]],
                               scale: float,
                               position: Tuple[float, float, float]) -> bool:
        """Internal method to add model from a mapper."""
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Apply color if provided
        if color is not None:
            actor.GetProperty().SetColor(color[0], color[1], color[2])
        else:
            # Random color
            actor.GetProperty().SetColor(np.random.rand(), np.random.rand(), np.random.rand())
            
        # Apply transformation
        transform = vtk.vtkTransform()
        transform.Scale(scale, scale, scale)
        transform.Translate(position[0], position[1], position[2])
        actor.SetUserTransform(transform)
        
        # Add to renderer and store
        self.renderer.AddActor(actor)
        self.models[name] = actor
        
        return True
    
    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the scene.
        
        Args:
            name: Name of the model to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        if name in self.models:
            self.renderer.RemoveActor(self.models[name])
            del self.models[name]
            if name in self.model_sources:
                del self.model_sources[name]
            return True
        return False
    
    def set_model_scale(self, name: str, scale: float) -> bool:
        """
        Set model scale (uniform scaling).
        
        Args:
            name: Model name
            scale: Scale factor
            
        Returns:
            bool: True if successful, False if model not found
        """
        if name not in self.models:
            print(f"Model '{name}' not found.")
            return False
            
        actor = self.models[name]
        current_transform = actor.GetUserTransform()
        if current_transform:
            # Get current position
            position = current_transform.GetPosition()
            # Create new transform with new scale
            new_transform = vtk.vtkTransform()
            new_transform.Scale(scale, scale, scale)
            new_transform.Translate(position[0], position[1], position[2])
            actor.SetUserTransform(new_transform)
        else:
            transform = vtk.vtkTransform()
            transform.Scale(scale, scale, scale)
            actor.SetUserTransform(transform)
        return True
    
    def set_model_pose(self, name: str, 
                      translation_matrix: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
                      rotation_matrix: Optional[np.ndarray] = None) -> bool:
        """
        Set model pose using translation and rotation matrices.
        
        Args:
            name: Model name
            translation_matrix: 3-element array for translation or 3x3/4x4 matrix
            rotation_matrix: 3x3 or 4x4 numpy array for rotation
            
        Returns:
            bool: True if successful, False if model not found
        """
        if name not in self.models:
            print(f"Model '{name}' not found.")
            return False
            
        actor = self.models[name]
        transform = vtk.vtkTransform()
        transform.Identity()
        
        # Apply rotation
        if rotation_matrix is not None:
            # Convert numpy array to vtk matrix
            if rotation_matrix.shape == (3, 3):
                mat = vtk.vtkMatrix4x4()
                mat.Identity()
                for i in range(3):
                    for j in range(3):
                        mat.SetElement(i, j, rotation_matrix[i, j])
                transform.Concatenate(mat)
            elif rotation_matrix.shape == (4, 4):
                mat = vtk.vtkMatrix4x4()
                for i in range(4):
                    for j in range(4):
                        mat.SetElement(i, j, rotation_matrix[i, j])
                transform.Concatenate(mat)
                
        # Apply translation
        if translation_matrix is not None:
            if isinstance(translation_matrix, tuple) and len(translation_matrix) == 3:
                transform.Translate(translation_matrix[0], translation_matrix[1], translation_matrix[2])
            elif isinstance(translation_matrix, np.ndarray):
                if translation_matrix.shape == (3,):
                    transform.Translate(translation_matrix[0], translation_matrix[1], translation_matrix[2])
                elif translation_matrix.shape == (3, 1):
                    transform.Translate(translation_matrix[0, 0], translation_matrix[1, 0], translation_matrix[2, 0])
                
        actor.SetUserTransform(transform)
        return True
    
    def set_model_position(self, name: str, x: float, y: float, z: float) -> bool:
        """
        Set model position.
        
        Args:
            name: Model name
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            bool: True if successful, False if model not found
        """
        if name not in self.models:
            print(f"Model '{name}' not found.")
            return False
            
        actor = self.models[name]
        current_transform = actor.GetUserTransform()
        if current_transform:
            # Get current scale
            scale = current_transform.GetScale()
            new_transform = vtk.vtkTransform()
            new_transform.Scale(scale[0], scale[1], scale[2])
            new_transform.Translate(x, y, z)
            actor.SetUserTransform(new_transform)
        else:
            transform = vtk.vtkTransform()
            transform.Translate(x, y, z)
            actor.SetUserTransform(transform)
        return True
    
    def get_model_list(self) -> List[str]:
        """Return list of all model names."""
        return list(self.models.keys())
    
    def clear_all_models(self) -> None:
        """Remove all models from the scene."""
        for name in list(self.models.keys()):
            self.remove_model(name)

class VTKWidget(QFrame):
    """VTK widget for 3D visualization with model management."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("background-color: #2a2a2a;")
        
        # Create VTK renderer and widget
        self.vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor(self)
        self.renderer: vtk.vtkRenderer = vtk.vtkRenderer()
        
        # IMPORTANT: Get the render window and configure it properly
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.renderer)
        
        # CRITICAL: Set window size and position to match parent widget
        # This prevents VTK from creating its own separate window
        render_window.SetSize(self.width(), self.height())
        render_window.SetPosition(0, 0)
        
        # Optional: Disable off-screen rendering if not needed
        # render_window.SetOffScreenRendering(False)
        
        # Set background color (dark gray)
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        
        # Create model manager
        self.model_manager: VTKModelManager = VTKModelManager(self.renderer)
        
        # Initialize camera with default settings
        self.set_camera_default()
        
        # Initialize interactor
        render_window.Render()
        self.vtk_widget.Initialize()
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        
        # Mouse interaction enabled by default
        self.mouse_interaction_enabled: bool = True
        
    # Override resize event to update VTK window size
    def resizeEvent(self, a0):
        """Update VTK render window size when widget is resized."""
        super().resizeEvent(a0)
        if hasattr(self, 'vtk_widget'):
            render_window = self.vtk_widget.GetRenderWindow()
            render_window.SetSize(self.width(), self.height())
            self.vtk_render()

    def set_camera_default(self) -> None:
        """Set default camera position."""
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().SetClippingRange(0.01, 10 ** 7)
        
    def set_camera_y_direction(self, up_vector: Tuple[float, float, float] = (0, 1, 0)) -> None:
        """
        Set camera's Y-axis direction (up vector).
        
        Args:
            up_vector: Tuple (x, y, z) specifying the up direction
        """
        self.renderer.GetActiveCamera().SetViewUp(up_vector[0], up_vector[1], up_vector[2])
        self.vtk_render()
        
    def set_camera_pose(self, position: Tuple[float, float, float], 
                       focal_point: Tuple[float, float, float], 
                       view_up: Tuple[float, float, float] = (0, 1, 0)) -> None:
        """
        Set camera position and orientation.
        
        Args:
            position: Tuple (x, y, z) camera position
            focal_point: Tuple (x, y, z) point the camera looks at
            view_up: Tuple (x, y, z) up vector
        """
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(position[0], position[1], position[2])
        camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
        camera.SetViewUp(view_up[0], view_up[1], view_up[2])
        self.vtk_render()
        
    def set_camera_clipping_range(self, near_plane: float = 0.01, far_plane: float = 1000) -> None:
        """
        Set camera clipping planes to maximum range.
        
        Args:
            near_plane: Near clipping plane distance
            far_plane: Far clipping plane distance
        """
        self.renderer.GetActiveCamera().SetClippingRange(near_plane, far_plane)
        self.vtk_render()

    def set_mouse_interaction(self, enabled: bool) -> None:
        """
        Enable or disable mouse interaction with the scene.
        
        Args:
            enabled: Boolean, True to enable mouse interaction, False to disable
        """
        self.mouse_interaction_enabled = enabled
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        if enabled:
            if hasattr(self, '_original_style'):
                interactor.SetInteractorStyle(self._original_style)
            else:
                style = vtk.vtkInteractorStyleTrackballCamera()
                interactor.SetInteractorStyle(style)
        else:
            # 保存当前样式
            self._original_style = interactor.GetInteractorStyle()
            null_style = vtk.vtkInteractorStyle()
            null_style.SetAutoAdjustCameraClippingRange(False)
            interactor.SetInteractorStyle(null_style)
        
        self.vtk_render()
            
    def add_model_from_file(self, name: str, file_path: str,
                           color: Optional[Tuple[float, float, float]] = None,
                           scale: float = 1.0,
                           position: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        """Add model from file."""
        return self.model_manager.add_model_from_file(name, file_path, color, scale, position)
    
    def remove_model(self, name: str) -> bool:
        """Remove a model by name."""
        return self.model_manager.remove_model(name)
    
    def set_model_scale(self, name: str, scale: float) -> bool:
        """Set model scale."""
        return self.model_manager.set_model_scale(name, scale)
    
    def set_model_pose(self, name: str,
                      translation_matrix: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
                      rotation_matrix: Optional[np.ndarray] = None) -> bool:
        """Set model pose with matrices."""
        return self.model_manager.set_model_pose(name, translation_matrix, rotation_matrix)
    
    def set_model_position(self, name: str, x: float, y: float, z: float) -> bool:
        """Set model position."""
        return self.model_manager.set_model_position(name, x, y, z)
    
    def get_model_list(self) -> List[str]:
        """Get list of all model names."""
        return self.model_manager.get_model_list()
    
    def clear_all_models(self) -> None:
        """Remove all models."""
        self.model_manager.clear_all_models()
        
    def vtk_render(self) -> None:
        """Force a render."""
        if self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()
            
    def get_renderer(self) -> vtk.vtkRenderer:
        """Return the VTK renderer for external manipulation."""
        return self.renderer


class ImageDisplayWidget(QFrame):
    """Widget for displaying a 2D slice from a 3D volume."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("background-color: black;")
        self.image_data: Optional[np.ndarray] = None
        self.setMinimumSize(100, 100)
        
    def update_slice(self, slice_2d: np.ndarray) -> None:
        """
        Update displayed 2D slice.
        
        Args:
            slice_2d: 2D numpy array (grayscale, uint8)
        """
        if slice_2d is None:
            return
        
        # Store a copy to avoid memory issues
        self.image_data = slice_2d.copy()
        self.update()  # trigger repaint
        
    def paintEvent(self, a0: QPaintEvent|None) -> None:
        """Paint the image scaled to fit the widget while preserving aspect ratio."""
        if self.image_data is None or self.image_data.size == 0:
            return
            
        h, w = self.image_data.shape
        # Create a copy to ensure data persists
        img_copy = self.image_data.copy()
        qimage = QImage(img_copy.data, w, h, w, QImage.Format_Grayscale8)
        
        # Scale to widget size while preserving aspect ratio
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation)
        
        # Center the image
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()


class VolumeSliceViewer:
    """Manages 3D volume and extracts slices for display."""
    
    def __init__(self, image_widgets: List[ImageDisplayWidget]):
        """
        Initialize the volume slice viewer.
        
        Args:
            image_widgets: List of 3 ImageDisplayWidgets for XY, XZ, YZ slices
        """
        self.volume_data: Optional[np.ndarray] = None  # 3D uint8 array
        self.image_widgets = image_widgets
        
        # Current slice indices
        self.current_x: int = 0
        self.current_y: int = 0
        self.current_z: int = 0
        
    def set_volume(self, volume: np.ndarray) -> None:
        """
        Set the 3D volume data.
        
        Args:
            volume: 3D numpy array (uint8) with shape (depth, height, width) or (z, y, x)
        """
        if volume is None:
            return
        
        assert volume.ndim == 3, "Volume must be 3D"
        assert volume.dtype == np.uint8, "Volume must be uint8"
        
        self.volume_data = volume
        
        # Update all slices with current indices
        self.update_all_slices()
    
    def set_slice_position(self, axis: str, position: int) -> None:
        """
        Set the slice position for a specific axis.
        
        Args:
            axis: 'x', 'y', or 'z'
            position: Slice index
        """
        if self.volume_data is None:
            return
        
        old_position = None
        if axis == 'x':
            old_position = self.current_x
            self.current_x = max(0, min(position, self.volume_data.shape[0] - 1))
            if old_position == self.current_x:
                return
        elif axis == 'y':
            old_position = self.current_y
            self.current_y = max(0, min(position, self.volume_data.shape[1] - 1))
            if old_position == self.current_y:
                return
        elif axis == 'z':
            old_position = self.current_z
            self.current_z = max(0, min(position, self.volume_data.shape[2] - 1))
            if old_position == self.current_z:
                return
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        # Update the corresponding slice
        self._update_slice(axis)
    
    def set_slice_positions(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        """
        Set multiple slice positions at once.
        
        Args:
            x: X slice index (optional)
            y: Y slice index (optional)
            z: Z slice index (optional)
        """
        if self.volume_data is None:
            return
        
        need_update = False
        
        if x is not None:
            new_x = max(0, min(x, self.volume_data.shape[0] - 1))
            if new_x != self.current_x:
                self.current_x = new_x
                need_update = True
        
        if y is not None:
            new_y = max(0, min(y, self.volume_data.shape[1] - 1))
            if new_y != self.current_y:
                self.current_y = new_y
                need_update = True
        
        if z is not None:
            new_z = max(0, min(z, self.volume_data.shape[2] - 1))
            if new_z != self.current_z:
                self.current_z = new_z
                need_update = True
        
        if need_update:
            self.update_all_slices()
    
    def update_all_slices(self) -> None:
        """Update all three slice views."""
        if self.volume_data is None:
            return
        
        # Extract and display XY slice (constant Z)
        xy_slice = self.volume_data[self.current_z, :, :]
        self.image_widgets[0].update_slice(xy_slice)
        
        # Extract and display XZ slice (constant Y)
        xz_slice = self.volume_data[:, self.current_y, :]
        self.image_widgets[1].update_slice(xz_slice)
        
        # Extract and display YZ slice (constant X)
        yz_slice = self.volume_data[:, :, self.current_x]
        self.image_widgets[2].update_slice(yz_slice)
    
    def _update_slice(self, axis: str) -> None:
        """Update a single slice based on axis."""
        if self.volume_data is None:
            return
        
        if axis == 'z':
            # XY slice
            xy_slice = self.volume_data[self.current_z, :, :]
            self.image_widgets[0].update_slice(xy_slice)
        elif axis == 'y':
            # XZ slice
            xz_slice = self.volume_data[:, self.current_y, :]
            self.image_widgets[1].update_slice(xz_slice)
        elif axis == 'x':
            # YZ slice
            yz_slice = self.volume_data[:, :, self.current_x]
            self.image_widgets[2].update_slice(yz_slice)
    
    def get_current_positions(self) -> Tuple[int, int, int]:
        """Get current slice positions (x, y, z)."""
        return (self.current_x, self.current_y, self.current_z)


class MedScopeWindow(QMainWindow):
    """Main window with four equal sub-windows."""
    
    def set_window_title(self, new_title:str, force:bool=False):
        if force or (new_title != self.window_title):
            self.window_title = new_title
            self.setWindowTitle(self.window_title)

    def __init__(self):
        super().__init__()

        # Set window title
        self.window_title = "MedScope"
        self.set_window_title(self.window_title, True)
        
        # Get screen resolution and set window to full screen
        screen = QApplication.primaryScreen()
        if screen is not None:
            screen_geometry = screen.availableGeometry()
            self.setGeometry(screen_geometry)
        else:
            self.setGeometry(0, 0, 1920, 1080)  # default size
        self.showMaximized()
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        
        # Create a 2x2 grid layout
        grid_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        
        # Top-left: VTK scene
        self.vtk_widget: VTKWidget = VTKWidget()
        left_column.addWidget(self.vtk_widget)
        
        # Bottom-left: XY slice (Z = constant)
        self.image_widget_xy: ImageDisplayWidget = ImageDisplayWidget()
        left_column.addWidget(self.image_widget_xy)
        
        # Top-right: XZ slice (Y = constant)
        self.image_widget_xz: ImageDisplayWidget = ImageDisplayWidget()
        right_column.addWidget(self.image_widget_xz)
        
        # Bottom-right: YZ slice (X = constant)
        self.image_widget_yz: ImageDisplayWidget = ImageDisplayWidget()
        right_column.addWidget(self.image_widget_yz)
        
        # Set equal stretch factors
        left_column.setStretch(0, 1)
        left_column.setStretch(1, 1)
        right_column.setStretch(0, 1)
        right_column.setStretch(1, 1)
        
        grid_layout.addLayout(left_column, 1)
        grid_layout.addLayout(right_column, 1)
        main_layout.addLayout(grid_layout)
        
        # Create volume slice viewer
        self.slice_viewer = VolumeSliceViewer([
            self.image_widget_xy,  # XY slice (Z constant)
            self.image_widget_xz,  # XZ slice (Y constant)
            self.image_widget_yz   # YZ slice (X constant)
        ])
        
        # Store all timers
        self.timer_pool: Dict[str, QTimer] = dict()
        
        # Default: ban mouse interaction
        self.set_mouse_interaction(False)
        
        # Create a default test volume
        self._create_default_volume()

        # Show window
        self.show()
    
    def _create_default_volume(self) -> None:
        """Create a fast test 3D volume with simple patterns."""
        self.set_volume(np.zeros((1, 1, 1), np.uint8))
    
    def add_timer(self, timer_name: str, timer_ms:int, call_func:Optional[Callable]=None) -> None:
        """Add or erase a timer to the pool."""
        assert timer_ms >= 0

        if self.timer_pool.get(timer_name) is not None:
            self.timer_pool[timer_name].stop()
            del self.timer_pool[timer_name]
        
        if call_func is not None:
            qtimer_now = QTimer()
            qtimer_now.timeout.connect(call_func)
            qtimer_now.start()
            self.timer_pool[timer_name] = qtimer_now
        
    def __del__(self):
        for timer_name in self.timer_pool:
            self.timer_pool[timer_name].stop()
    
    # Volume slice interface methods
    def set_volume(self, volume: np.ndarray) -> None:
        """
        Set the 3D volume data.
        
        Args:
            volume: 3D numpy array (uint8) with shape (z, y, x)
        """
        self.slice_viewer.set_volume(volume)
    
    def set_slice_x(self, x: int) -> None:
        """Set the X slice position (for YZ plane)."""
        self.slice_viewer.set_slice_position('x', x)
    
    def set_slice_y(self, y: int) -> None:
        """Set the Y slice position (for XZ plane)."""
        self.slice_viewer.set_slice_position('y', y)
    
    def set_slice_z(self, z: int) -> None:
        """Set the Z slice position (for XY plane)."""
        self.slice_viewer.set_slice_position('z', z)
    
    def set_slice_positions(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        """Set multiple slice positions at once."""
        self.slice_viewer.set_slice_positions(x, y, z)
    
    def get_slice_positions(self) -> Tuple[int, int, int]:
        """Get current slice positions (x, y, z)."""
        return self.slice_viewer.get_current_positions()
    
    # VTK interface methods for external use
    def get_vtk_renderer(self) -> vtk.vtkRenderer:
        """Return the VTK renderer."""
        return self.vtk_widget.get_renderer()
        
    def set_camera_y_direction(self, up_vector: Tuple[float, float, float] = (0, 1, 0)) -> None:
        """Set camera's up direction."""
        self.vtk_widget.set_camera_y_direction(up_vector)
        
    def set_camera_pose(self, position: Tuple[float, float, float], 
                       focal_point: Tuple[float, float, float], 
                       view_up: Tuple[float, float, float] = (0, 1, 0)) -> None:
        """Set camera position and orientation."""
        self.vtk_widget.set_camera_pose(position, focal_point, view_up)
        
    def set_camera_clipping_range(self, near_plane: float = 0.01, far_plane: float = 10**7) -> None:
        """Set camera clipping range."""
        self.vtk_widget.set_camera_clipping_range(near_plane, far_plane)
        
    def set_mouse_interaction(self, enabled: bool) -> None:
        """Enable/disable mouse interaction."""
        self.vtk_widget.set_mouse_interaction(enabled)
        
    def add_model_from_file(self, name: str, file_path: str,
                           color: Optional[Tuple[float, float, float]] = None,
                           scale: float = 1.0,
                           position: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        """Add 3D model from file."""
        return self.vtk_widget.add_model_from_file(name, file_path, color, scale, position)
        
    def remove_model(self, name: str) -> bool:
        """Remove a model by name."""
        return self.vtk_widget.remove_model(name)
        
    def set_model_position(self, name: str, x: float, y: float, z: float) -> bool:
        """Set model position."""
        ans = self.vtk_widget.set_model_position(name, x, y, z)
        self.vtk_widget.vtk_render()
        return ans
    
    def set_model_scale(self, name: str, scale: float) -> bool:
        """Set model scale."""
        ans = self.vtk_widget.set_model_scale(name, scale)
        self.vtk_widget.vtk_render()
        return ans
    
    def set_model_pose(self, name: str,
                      translation_matrix: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
                      rotation_matrix: Optional[np.ndarray] = None) -> bool:
        """Set model pose with matrices."""
        ans = self.vtk_widget.set_model_pose(name, translation_matrix, rotation_matrix)
        self.vtk_widget.vtk_render()
        return ans
        
    def get_model_list(self) -> List[str]:
        """Get list of all model names."""
        return self.vtk_widget.get_model_list()
        
    def clear_all_models(self) -> None:
        """Remove all models."""
        self.vtk_widget.clear_all_models()
