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
        self.renderer: vtk.vtkRenderer = renderer
        self.models: Dict[str, vtk.vtkActor] = {}
        self.model_sources: Dict[str, Any] = {}
        
    def add_model_from_file(self, name: str, file_path: str, 
                           color: Optional[Tuple[float, float, float]] = None, 
                           scale: float = 1.0, 
                           position: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        if name in self.models:
            print(f"Model '{name}' already exists. Remove it first or use a different name.")
            return False
            
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
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(algorithm.GetOutputPort())
        return self._add_model_from_mapper(name, mapper, color, scale, position)
    
    def _add_model_from_mapper(self, name: str, mapper: vtk.vtkPolyDataMapper,
                               color: Optional[Tuple[float, float, float]],
                               scale: float,
                               position: Tuple[float, float, float]) -> bool:
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        if color is not None:
            actor.GetProperty().SetColor(color[0], color[1], color[2])
        else:
            actor.GetProperty().SetColor(np.random.rand(), np.random.rand(), np.random.rand())
            
        transform = vtk.vtkTransform()
        transform.Scale(scale, scale, scale)
        transform.Translate(position[0], position[1], position[2])
        actor.SetUserTransform(transform)
        
        self.renderer.AddActor(actor)
        self.models[name] = actor
        
        return True
    
    def remove_model(self, name: str) -> bool:
        if name in self.models:
            self.renderer.RemoveActor(self.models[name])
            del self.models[name]
            if name in self.model_sources:
                del self.model_sources[name]
            return True
        return False
    
    def set_model_scale(self, name: str, scale: float) -> bool:
        if name not in self.models:
            print(f"Model '{name}' not found.")
            return False
            
        actor = self.models[name]
        current_transform = actor.GetUserTransform()
        if current_transform:
            position = current_transform.GetPosition()
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
        if name not in self.models:
            print(f"Model '{name}' not found.")
            return False
            
        actor = self.models[name]
        transform = vtk.vtkTransform()
        transform.Identity()
        
        if rotation_matrix is not None:
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
        if name not in self.models:
            print(f"Model '{name}' not found.")
            return False
            
        actor = self.models[name]
        current_transform = actor.GetUserTransform()
        if current_transform:
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
        return list(self.models.keys())
    
    def clear_all_models(self) -> None:
        for name in list(self.models.keys()):
            self.remove_model(name)

class VTKWidget(QFrame):
    """VTK widget for 3D visualization with model management."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("background-color: #2a2a2a;")
        
        self.vtk_widget: QVTKRenderWindowInteractor = QVTKRenderWindowInteractor(self)
        self.renderer: vtk.vtkRenderer = vtk.vtkRenderer()
        
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.renderer)
        render_window.SetSize(self.width(), self.height())
        render_window.SetPosition(0, 0)
        
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.model_manager: VTKModelManager = VTKModelManager(self.renderer)
        self.set_camera_default()
        
        render_window.Render()
        self.vtk_widget.Initialize()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        
        self.mouse_interaction_enabled: bool = True
        
    def resizeEvent(self, a0):
        super().resizeEvent(a0)
        if hasattr(self, 'vtk_widget'):
            render_window = self.vtk_widget.GetRenderWindow()
            render_window.SetSize(self.width(), self.height())
            self.vtk_render()

    def set_camera_default(self) -> None:
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().SetClippingRange(0.01, 10 ** 7)
        
    def set_camera_y_direction(self, up_vector: Tuple[float, float, float] = (0, 1, 0)) -> None:
        self.renderer.GetActiveCamera().SetViewUp(up_vector[0], up_vector[1], up_vector[2])
        self.vtk_render()
        
    def set_camera_pose(self, position: Tuple[float, float, float], 
                       focal_point: Tuple[float, float, float], 
                       view_up: Tuple[float, float, float] = (0, 1, 0)) -> None:
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(position[0], position[1], position[2])
        camera.SetFocalPoint(focal_point[0], focal_point[1], focal_point[2])
        camera.SetViewUp(view_up[0], view_up[1], view_up[2])
        self.vtk_render()
        
    def set_camera_clipping_range(self, near_plane: float = 0.01, far_plane: float = 1000) -> None:
        self.renderer.GetActiveCamera().SetClippingRange(near_plane, far_plane)
        self.vtk_render()

    def set_mouse_interaction(self, enabled: bool) -> None:
        self.mouse_interaction_enabled = enabled
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        if enabled:
            if hasattr(self, '_original_style'):
                interactor.SetInteractorStyle(self._original_style)
            else:
                style = vtk.vtkInteractorStyleTrackballCamera()
                interactor.SetInteractorStyle(style)
        else:
            self._original_style = interactor.GetInteractorStyle()
            null_style = vtk.vtkInteractorStyle()
            null_style.SetAutoAdjustCameraClippingRange(False)
            interactor.SetInteractorStyle(null_style)
        
        self.vtk_render()
            
    def add_model_from_file(self, name: str, file_path: str,
                           color: Optional[Tuple[float, float, float]] = None,
                           scale: float = 1.0,
                           position: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        return self.model_manager.add_model_from_file(name, file_path, color, scale, position)
    
    def remove_model(self, name: str) -> bool:
        return self.model_manager.remove_model(name)
    
    def set_model_scale(self, name: str, scale: float) -> bool:
        return self.model_manager.set_model_scale(name, scale)
    
    def set_model_pose(self, name: str,
                      translation_matrix: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
                      rotation_matrix: Optional[np.ndarray] = None) -> bool:
        return self.model_manager.set_model_pose(name, translation_matrix, rotation_matrix)
    
    def set_model_position(self, name: str, x: float, y: float, z: float) -> bool:
        return self.model_manager.set_model_position(name, x, y, z)
    
    def get_model_list(self) -> List[str]:
        return self.model_manager.get_model_list()
    
    def clear_all_models(self) -> None:
        self.model_manager.clear_all_models()
        
    def vtk_render(self) -> None:
        if self.vtk_widget:
            self.vtk_widget.GetRenderWindow().Render()
            
    def get_renderer(self) -> vtk.vtkRenderer:
        return self.renderer

class ImageDisplayWidget(QFrame):
    """RGB 彩色图像显示控件，适配 3 通道切片"""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("background-color: black;")
        self.image_data: Optional[np.ndarray] = None
        self.setMinimumSize(100, 100)
        
    def update_slice(self, slice_rgb: np.ndarray) -> None:
        """
        更新显示 RGB 2D 切片
        Args:
            slice_rgb: shape=(H, W, 3), dtype=np.uint8
        """
        if slice_rgb is None:
            return
        self.image_data = slice_rgb.copy()
        self.update()
        
    def paintEvent(self, a0: QPaintEvent|None) -> None:
        if self.image_data is None or self.image_data.size == 0:
            return
            
        h, w, _ = self.image_data.shape
        img_copy = self.image_data.copy()
        # RGB 格式显示
        qimage = QImage(img_copy.data, w, h, 3 * w, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation)
        
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)
        painter.end()

class VolumeSliceViewer:
    """适配 4D RGB 3D 图像 (3, N, M, L) 的切片管理器"""
    
    def __init__(self, image_widgets: List[ImageDisplayWidget]):
        self.volume_data: Optional[np.ndarray] = None  # shape=(3, N, M, L), uint8
        self.image_widgets = image_widgets
        
        self.current_x: int = 0
        self.current_y: int = 0
        self.current_z: int = 0
        
    def set_volume(self, volume: np.ndarray) -> None:
        """
        设置 RGB 3D 图像
        Args:
            volume: 4D numpy array, shape=(3, N, M, L), dtype=np.uint8
        """
        if volume is None:
            return
        
        # Map 3D image to 3 channel color image
        if volume.ndim == 3:
            volume = np.repeat(volume[np.newaxis, ...], 3, axis=0)
        
        if volume.ndim != 4:
            raise ValueError("volume must be 3D or 4D array")
        
        if volume.shape[0] != 3:
            raise ValueError("First dimension of valume must be 3 (channels)")
        
        if volume.dtype != np.uint8:
            raise ValueError("dtype of volume must be numpy.uint8")
        
        self.volume_data = volume
        _, h, w, d = volume.shape
        self.current_x = h // 2
        self.current_y = w // 2
        self.current_z = d // 2
        self.update_all_slices()
    
    def set_slice_position(self, axis: str, position: int) -> None:
        if self.volume_data is None:
            return
        
        _, h, w, d = self.volume_data.shape
        if axis == 'x':
            self.current_x = max(0, min(position, h - 1))
        elif axis == 'y':
            self.current_y = max(0, min(position, w - 1))
        elif axis == 'z':
            self.current_z = max(0, min(position, d - 1))
        else:
            raise ValueError("Axis must be 'x', 'y', 'z'")
        
        self._update_slice(axis)
    
    def set_slice_positions(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        if self.volume_data is None:
            return
        
        _, h, w, d = self.volume_data.shape
        if x is not None:
            self.current_x = max(0, min(x, h - 1))
        if y is not None:
            self.current_y = max(0, min(y, w - 1))
        if z is not None:
            self.current_z = max(0, min(z, d - 1))
        
        self.update_all_slices()
    
    def update_all_slices(self) -> None:
        if self.volume_data is None:
            return
        
        # XY 切面 (固定Z) → (3, H, W) → (H, W, 3)
        xy = self.volume_data[:, :, :, self.current_z].transpose(1, 2, 0)
        self.image_widgets[0].update_slice(xy)
        
        # XZ 切面 (固定Y) → (3, H, D) → (H, D, 3)
        xz = self.volume_data[:, :, self.current_y, :].transpose(1, 2, 0)
        self.image_widgets[1].update_slice(xz)
        
        # YZ 切面 (固定X) → (3, W, D) → (W, D, 3)
        yz = self.volume_data[:, self.current_x, :, :].transpose(1, 2, 0)
        self.image_widgets[2].update_slice(yz)
    
    def _update_slice(self, axis: str) -> None:
        if self.volume_data is None:
            return
        
        if axis == 'z':
            xy = self.volume_data[:, :, :, self.current_z].transpose(1, 2, 0)
            self.image_widgets[0].update_slice(xy)
        elif axis == 'y':
            xz = self.volume_data[:, :, self.current_y, :].transpose(1, 2, 0)
            self.image_widgets[1].update_slice(xz)
        elif axis == 'x':
            yz = self.volume_data[:, self.current_x, :, :].transpose(1, 2, 0)
            self.image_widgets[2].update_slice(yz)
    
    def get_current_positions(self) -> Tuple[int, int, int]:
        return self.current_x, self.current_y, self.current_z

class MedScopeWindow(QMainWindow):
    """主窗口：完全兼容 RGB 4D 输入，功能不变"""
    
    def set_window_title(self, new_title:str, force:bool=False):
        if force or (new_title != self.window_title):
            self.window_title = new_title
            self.setWindowTitle(self.window_title)

    def __init__(self):
        super().__init__()

        self.window_title = "MedScope RGB 3D图像查看器"
        self.set_window_title(self.window_title, True)
        
        screen = QApplication.primaryScreen()
        if screen is not None:
            screen_geometry = screen.availableGeometry()
            self.setGeometry(screen_geometry)
        else:
            self.setGeometry(0, 0, 1920, 1080)
        self.showMaximized()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        
        grid_layout = QHBoxLayout()
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        
        self.vtk_widget: VTKWidget = VTKWidget()
        left_column.addWidget(self.vtk_widget)
        
        self.image_widget_xy: ImageDisplayWidget = ImageDisplayWidget()
        left_column.addWidget(self.image_widget_xy)
        
        self.image_widget_xz: ImageDisplayWidget = ImageDisplayWidget()
        right_column.addWidget(self.image_widget_xz)
        
        self.image_widget_yz: ImageDisplayWidget = ImageDisplayWidget()
        right_column.addWidget(self.image_widget_yz)
        
        left_column.setStretch(0, 1)
        left_column.setStretch(1, 1)
        right_column.setStretch(0, 1)
        right_column.setStretch(1, 1)
        
        grid_layout.addLayout(left_column, 1)
        grid_layout.addLayout(right_column, 1)
        main_layout.addLayout(grid_layout)
        
        self.slice_viewer = VolumeSliceViewer([
            self.image_widget_xy,
            self.image_widget_xz,
            self.image_widget_yz
        ])
        
        self.timer_pool: Dict[str, QTimer] = dict()
        self.set_mouse_interaction(False)
        self._create_default_volume()
        self.show()
    
    def _create_default_volume(self) -> None:
        """生成默认 RGB 3D 测试数据 (3, 64, 64, 64)"""
        rgb_volume = np.zeros((3, 64, 64, 64), dtype=np.uint8)
        rgb_volume[0] = 128  # R通道
        rgb_volume[1] = 128  # G通道
        rgb_volume[2] = 128  # B通道
        self.set_volume(rgb_volume)

    def add_timer(self, timer_name: str, timer_ms:int, call_func:Optional[Callable]=None) -> None:
        assert timer_ms >= 0
        if self.timer_pool.get(timer_name) is not None:
            self.timer_pool[timer_name].stop()
            del self.timer_pool[timer_name]
        
        if call_func is not None:
            qtimer_now = QTimer()
            qtimer_now.timeout.connect(call_func)
            qtimer_now.start(timer_ms)
            self.timer_pool[timer_name] = qtimer_now
        
    def __del__(self):
        for timer_name in self.timer_pool:
            self.timer_pool[timer_name].stop()
    
    # 对外接口：输入 4D RGB 数组 (3, N, M, L)
    def set_volume(self, volume: np.ndarray) -> None:
        self.slice_viewer.set_volume(volume)
    
    def set_slice_x(self, x: int) -> None:
        self.slice_viewer.set_slice_position('x', x)
    
    def set_slice_y(self, y: int) -> None:
        self.slice_viewer.set_slice_position('y', y)
    
    def set_slice_z(self, z: int) -> None:
        self.slice_viewer.set_slice_position('z', z)
    
    def set_slice_positions(self, x: Optional[int] = None, y: Optional[int] = None, z: Optional[int] = None) -> None:
        self.slice_viewer.set_slice_positions(x, y, z)
    
    def get_slice_positions(self) -> Tuple[int, int, int]:
        return self.slice_viewer.get_current_positions()
    
    # VTK 接口完全不变
    def get_vtk_renderer(self) -> vtk.vtkRenderer:
        return self.vtk_widget.get_renderer()
        
    def set_camera_y_direction(self, up_vector: Tuple[float, float, float] = (0, 1, 0)) -> None:
        self.vtk_widget.set_camera_y_direction(up_vector)
        
    def set_camera_pose(self, position: Tuple[float, float, float], 
                       focal_point: Tuple[float, float, float], 
                       view_up: Tuple[float, float, float] = (0, 1, 0)) -> None:
        self.vtk_widget.set_camera_pose(position, focal_point, view_up)
        
    def set_camera_clipping_range(self, near_plane: float = 0.01, far_plane: float = 10**7) -> None:
        self.vtk_widget.set_camera_clipping_range(near_plane, far_plane)
        
    def set_mouse_interaction(self, enabled: bool) -> None:
        self.vtk_widget.set_mouse_interaction(enabled)
        
    def add_model_from_file(self, name: str, file_path: str,
                           color: Optional[Tuple[float, float, float]] = None,
                           scale: float = 1.0,
                           position: Tuple[float, float, float] = (0, 0, 0)) -> bool:
        return self.vtk_widget.add_model_from_file(name, file_path, color, scale, position)
        
    def remove_model(self, name: str) -> bool:
        return self.vtk_widget.remove_model(name)
        
    def set_model_position(self, name: str, x: float, y: float, z: float) -> bool:
        ans = self.vtk_widget.set_model_position(name, x, y, z)
        self.vtk_widget.vtk_render()
        return ans
    
    def set_model_scale(self, name: str, scale: float) -> bool:
        ans = self.vtk_widget.set_model_scale(name, scale)
        self.vtk_widget.vtk_render()
        return ans
    
    def set_model_pose(self, name: str,
                      translation_matrix: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
                      rotation_matrix: Optional[np.ndarray] = None) -> bool:
        ans = self.vtk_widget.set_model_pose(name, translation_matrix, rotation_matrix)
        self.vtk_widget.vtk_render()
        return ans
        
    def get_model_list(self) -> List[str]:
        return self.vtk_widget.get_model_list()
        
    def clear_all_models(self) -> None:
        self.vtk_widget.clear_all_models()
