from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor, read_image)

class AdaptiveStitcher:
    """
    自适应图像拼接器，维护一个可改变大小的坐标系
    """
    def __init__(self):
        # 全局坐标系中的当前拼接结果
        self.global_canvas = None
        # 当前坐标系的原点在全局坐标系中的位置
        self.current_origin = np.array([0, 0], dtype=np.float32)
        # 累积变换矩阵（从当前坐标系到全局坐标系）
        self.accumulated_transform = np.eye(3, dtype=np.float32)
        # 存储所有帧的变换信息
        self.frame_transforms = []
        # 用于强制保持原始尺寸的标志
        self.preserve_scale = True
    
    def initialize_with_first_frame(self, first_frame):
        """
        用第一帧初始化拼接器
        """
        self.global_canvas = first_frame.copy()
        self.current_origin = np.array([0, 0], dtype=np.float32)
        self.accumulated_transform = np.eye(3, dtype=np.float32)
        self.frame_transforms = [{'transform': np.eye(3), 'origin': np.array([0, 0])}]
        print("Initialized stitcher with first frame")
    
    def compute_rigid_transform(self, mkpts0, mkpts1):
        """
        计算从图像1到图像2的刚性变换，强制保持原始尺寸
        
        Args:
            mkpts0: 参考图像中的匹配点 (图像1)
            mkpts1: 当前图像中的匹配点 (图像2)
            
        Returns:
            transform: 3x3 变换矩阵 (从图像2到图像1)
            inlier_count: 内点数量 (整数)
        """
        if len(mkpts0) < 4:
            print(f"Not enough matches for transform estimation: {len(mkpts0)}")
            return None, 0
        
        try:
            # 方法1: 使用estimateAffinePartial2D（OpenCV 4.x+ 替代estimateRigidTransform）
            result = cv2.estimateAffinePartial2D(
                mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            
            # estimateAffinePartial2D 返回 (transform, inliers)
            transform = result[0]
            inliers = result[1]
            
            if transform is None:
                print("Affine partial 2D estimation failed, trying homography...")
                # 方法2: 使用单应性矩阵但强制保持尺度
                H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
                if H is not None:
                    # 从单应性矩阵提取刚性变换部分
                    transform = self._homography_to_rigid(H)
                    inlier_count = np.sum(mask) if mask is not None else len(mkpts0)
                else:
                    # 方法3: 使用最小二乘法计算相似变换
                    transform = self._compute_similarity_transform(mkpts1, mkpts0)
                    inlier_count = len(mkpts0) if transform is not None else 0
            else:
                # 计算内点数量
                inlier_count = np.sum(inliers) if inliers is not None else len(mkpts0)
            
            if transform is None:
                print("All transform estimation methods failed")
                return None, 0
                
            # 转换为 3x3 齐次坐标矩阵
            transform_3x3 = np.vstack([transform, [0, 0, 1]])
            
            # 强制保持原始尺寸：移除缩放分量
            if self.preserve_scale:
                transform_3x3 = self._remove_scale_from_transform(transform_3x3)
            
            print(f"Transform matrix:\n{transform_3x3}")
            print(f"Inlier count: {inlier_count}")
            
            return transform_3x3, inlier_count
            
        except Exception as e:
            print(f"Error in transform computation: {e}")
            import traceback
            traceback.print_exc()
            return None, 0
    
    def _compute_similarity_transform(self, src_points, dst_points):
        """
        使用最小二乘法计算相似变换（旋转、平移、均匀缩放）
        """
        if len(src_points) < 2:
            return None
            
        try:
            # 中心化点集
            src_centroid = np.mean(src_points, axis=0)
            dst_centroid = np.mean(dst_points, axis=0)
            
            src_centered = src_points - src_centroid
            dst_centered = dst_points - dst_centroid
            
            # 计算缩放和旋转
            src_norm = np.sum(src_centered ** 2)
            if src_norm == 0:
                return None
                
            # 计算相关系数矩阵
            A = np.dot(dst_centered.T, src_centered) / src_norm
            
            # 提取旋转和缩放
            U, s, Vt = np.linalg.svd(A)
            rotation = U @ Vt
            
            # 如果行列式为负，需要调整以保持方向
            if np.linalg.det(rotation) < 0:
                Vt[-1, :] *= -1
                rotation = U @ Vt
            
            # 构建变换矩阵
            transform = np.eye(3)
            transform[0:2, 0:2] = rotation
            transform[0:2, 2] = dst_centroid - np.dot(rotation, src_centroid)
            
            return transform[0:2, :]  # 返回2x3矩阵
            
        except Exception as e:
            print(f"Error in similarity transform: {e}")
            return None
    
    def _homography_to_rigid(self, H):
        """
        从单应性矩阵提取刚性变换（旋转和平移）
        """
        try:
            # 分解单应性矩阵得到旋转和平移
            # 这里使用简化的方法：只提取左上角的2x2矩阵作为旋转部分
            rotation = H[0:2, 0:2]
            
            # 对旋转矩阵进行SVD分解，然后重建为纯旋转矩阵
            U, s, Vt = np.linalg.svd(rotation)
            rotation_clean = U @ Vt
            
            # 构建刚性变换矩阵
            rigid_transform = np.eye(3)
            rigid_transform[0:2, 0:2] = rotation_clean
            rigid_transform[0:2, 2] = H[0:2, 2]
            
            return rigid_transform[0:2, :]  # 返回2x3矩阵
            
        except Exception as e:
            print(f"Error converting homography to rigid: {e}")
            return None
    
    def _remove_scale_from_transform(self, transform):
        """
        从变换矩阵中移除缩放分量，只保留旋转和平移
        """
        try:
            # 提取旋转部分
            rotation = transform[0:2, 0:2]
            
            # 计算当前缩放因子
            scale_x = np.linalg.norm(rotation[:, 0])
            scale_y = np.linalg.norm(rotation[:, 1])
            
            print(f"Detected scales - X: {scale_x:.4f}, Y: {scale_y:.4f}")
            
            # 如果缩放接近1，保持原样；否则归一化
            if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                # 归一化旋转矩阵
                rotation_normalized = rotation / np.sqrt(scale_x * scale_y)
                
                # 重建变换矩阵
                transform_clean = np.eye(3)
                transform_clean[0:2, 0:2] = rotation_normalized
                transform_clean[0:2, 2] = transform[0:2, 2]
                
                print("Removed scale from transform")
                return transform_clean
            else:
                return transform
                
        except Exception as e:
            print(f"Error removing scale: {e}")
            return transform
    
    def update_coordinate_system(self, new_transform, current_frame):
        """
        更新坐标系并将新帧拼接到全局画布上
        """
        if new_transform is None:
            return False
            
        try:
            # 更新累积变换：新帧到全局坐标系的变换
            updated_transform = self.accumulated_transform @ new_transform
            
            # 计算新帧在全局坐标系中的边界
            h, w = current_frame.shape[:2]
            corners = np.array([
                [0, 0, 1],
                [w, 0, 1], 
                [w, h, 1],
                [0, h, 1]
            ], dtype=np.float32).T
            
            # 变换角点到全局坐标系
            transformed_corners = updated_transform @ corners
            transformed_corners = transformed_corners[:2] / transformed_corners[2]
            transformed_corners = transformed_corners.T
            
            # 计算全局画布需要扩展的范围
            if self.global_canvas is None:
                # 第一次调用，直接用当前帧初始化
                self.global_canvas = current_frame.copy()
                self.accumulated_transform = new_transform.copy()
                return True
                
            canvas_h, canvas_w = self.global_canvas.shape[:2]
            
            # 当前画布在全局坐标系中的角点
            canvas_corners = np.array([
                [0, 0],
                [canvas_w, 0],
                [canvas_w, canvas_h],
                [0, canvas_h]
            ], dtype=np.float32)
            
            # 所有角点（当前画布 + 新帧）
            all_corners = np.vstack([canvas_corners, transformed_corners])
            
            # 计算新的边界
            x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
            
            # 确保最小尺寸不小于原始图像尺寸
            min_width = max(canvas_w, w)
            min_height = max(canvas_h, h)
            
            new_width = max(x_max - x_min, min_width)
            new_height = max(y_max - y_min, min_height)
            
            # 计算平移量以确保所有内容可见
            translation_x = max(0, -x_min)
            translation_y = max(0, -y_min)
            
            # 如果画布需要扩展
            if new_width > canvas_w or new_height > canvas_h:
                print(f"Expanding canvas from {canvas_w}x{canvas_h} to {new_width}x{new_height}")
                
                # 创建新的扩展画布
                new_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                
                # 将原画布内容复制到新位置
                start_x = translation_x
                start_y = translation_y
                end_x = start_x + canvas_w
                end_y = start_y + canvas_h
                
                new_canvas[start_y:end_y, start_x:end_x] = self.global_canvas
                self.global_canvas = new_canvas
                
                # 更新原点偏移
                self.current_origin += np.array([translation_x, translation_y])
            
            # 调整变换矩阵以考虑画布扩展
            adjustment_transform = np.eye(3, dtype=np.float32)
            adjustment_transform[0, 2] = translation_x
            adjustment_transform[1, 2] = translation_y
            
            final_transform = adjustment_transform @ updated_transform
            
            # 将当前帧变换到全局坐标系
            warped_frame = cv2.warpAffine(
                current_frame, 
                final_transform[:2], 
                (self.global_canvas.shape[1], self.global_canvas.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT
            )
            
            # 创建融合掩码
            mask = (warped_frame > 0).all(axis=2)
            
            # 简单融合：新帧覆盖旧内容（可以改为更复杂的融合方法）
            self.global_canvas[mask] = warped_frame[mask]
            
            # 更新累积变换（考虑画布调整）
            self.accumulated_transform = adjustment_transform @ updated_transform
            
            # 记录变换信息
            self.frame_transforms.append({
                'transform': final_transform.copy(),
                'origin': self.current_origin.copy()
            })
            
            return True
            
        except Exception as e:
            print(f"Error updating coordinate system: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_current_canvas(self):
        """返回当前拼接结果"""
        return self.global_canvas

def extract_matching_features(matching_model, image0_gray, image1_gray, device):
    """
    提取两幅图像之间的匹配特征
    """
    keys = ['keypoints', 'scores', 'descriptors']
    
    # 处理第一张图像
    frame_0 = frame2tensor(image0_gray, device)
    last_data = matching_model.superpoint({'image': frame_0})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_0

    # 处理第二张图像
    frame_1 = frame2tensor(image1_gray, device)
    pred = matching_model({**last_data, 'image1': frame_1})

    # 提取匹配点
    kpts0 = last_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    # 筛选有效匹配
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    return mkpts0, mkpts1, kpts0, kpts1, matches, confidence

def stitch_image_sequence_robust(image_dir, output_dir, image_ext='png', max_images=20, config=None):
    """
    改进的拼接函数：强制保持原始图像尺寸
    """
    # 设置设备
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running inference on device "{device}"')
    
    # 默认配置
    if config is None:
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1000
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.7,
            }
        }
    
    # 获取图像文件列表
    image_paths = sorted(Path(image_dir).glob(f'*.{image_ext}'))
    if not image_paths:
        print(f"No images found in {image_dir} with extension {image_ext}")
        return None
    
    image_paths = image_paths[:max_images]
    print(f"Found {len(image_paths)} images to stitch")
    
    # 初始化匹配模型
    matching = Matching(config).eval().to(device)
    
    # 初始化拼接器
    stitcher = AdaptiveStitcher()
    
    prev_frame = None
    prev_frame_gray = None
    successful_stitches = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path.name}")
        
        # 读取图像
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            # 第一帧，初始化拼接器
            stitcher.initialize_with_first_frame(frame)
            prev_frame = frame
            prev_frame_gray = frame_gray
            successful_stitches += 1
            continue
        
        # 提取匹配特征
        mkpts0, mkpts1, kpts0, kpts1, matches, confidence = extract_matching_features(
            matching, prev_frame_gray, frame_gray, device
        )
        
        print(f"Found {len(mkpts0)} matches between image {i} and {i+1}")
        
        if len(mkpts0) >= 4:
            # 计算变换矩阵
            transform, inlier_count = stitcher.compute_rigid_transform(mkpts0, mkpts1)
            
            # 确保 inlier_count 是标量而不是数组
            if hasattr(inlier_count, '__len__') and not isinstance(inlier_count, (int, float)):
                inlier_count = inlier_count[0] if len(inlier_count) > 0 else 0
            
            if transform is not None and inlier_count >= 4:
                # 更新坐标系并拼接
                success = stitcher.update_coordinate_system(transform, frame)
                
                if success:
                    successful_stitches += 1
                    print(f"Successfully stitched image {i+1} (total: {successful_stitches})")
                    
                    # 显示进度
                    if (i + 1) % 3 == 0:
                        current_canvas = stitcher.get_current_canvas()
                        # 调整显示大小以便查看
                        display_img = current_canvas.copy()
                        h, w = display_img.shape[:2]
                        if w > 1200:
                            scale = 1200.0 / w
                            new_w = 1200
                            new_h = int(h * scale)
                            display_img = cv2.resize(display_img, (new_w, new_h))
                        cv2.imshow('Stitching Progress', display_img)
                        cv2.waitKey(1)
            else:
                print(f"Transform estimation failed for image {i+1}")
        else:
            print(f"Not enough matches for image {i+1}")
        
        # 更新前一帧
        prev_frame = frame
        prev_frame_gray = frame_gray
    
    # 保存最终结果
    if stitcher.get_current_canvas() is not None and successful_stitches > 1:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        final_result = stitcher.get_current_canvas()
        output_file = str(output_path / 'improved_stitching_result.png')
        cv2.imwrite(output_file, final_result)
        print(f'\nFinal stitching result saved to: {output_file}')
        print(f'Final canvas size: {final_result.shape[1]}x{final_result.shape[0]}')
        print(f'Successfully stitched {successful_stitches} images out of {len(image_paths)}')
        
        # 显示最终结果
        display_final = final_result.copy()
        h, w = display_final.shape[:2]
        if w > 1200:
            scale = 1200.0 / w
            new_w = 1200
            new_h = int(h * scale)
            display_final = cv2.resize(display_final, (new_w, new_h))
        cv2.imshow('Final Stitching Result', display_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return final_result
    else:
        print(f"Stitching failed - only {successful_stitches} successful stitches")
        return None

# 使用示例
if __name__ == "__main__":
    image_dir = 'C:/Users/Chen/Desktop/SuperGluePretrainedNetwork-master/data_input_lot'
    output_dir = 'C:/Users/Chen/Desktop/SuperGluePretrainedNetwork-master/data_output'
    
    result = stitch_image_sequence_robust(image_dir, output_dir, image_ext='png', max_images=10)
    
    if result is not None:
        print('Multi-frame stitching completed successfully!')
        print(f'Final canvas size: {result.shape[1]}x{result.shape[0]}')
    else:
        print('Multi-frame stitching failed!')