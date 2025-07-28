from fr3_controller import FR3Controller
from camera import RS_D455

import rclpy
from rclpy.executors import MultiThreadedExecutor
# from roboexp.utils import rpy_to_rotation_matrix
from transforms3d.euler import euler2mat
import cv2
import time
import numpy as np
import pickle
import random
import math
from pathlib import Path
import json
from typing import List, Tuple, Sequence
from pupil_apriltags import Detector


def rpy_to_rotation_matrix(roll, pitch, yaw):
    
    return euler2mat(roll, pitch, yaw, axes="sxyz")


class RoboCalibrate:
    """
    Robot calibration envrionment.
    1. require robot to have following APIs:
            move_to(x:float, y:float, z:float, roll:float, pitch:float, yaw:float, execute:bool): 
            function that can control the robot to a targeted pose.
    2. Require realsense camera with pylrealsense2 installed
    """

    def __init__(self):

        rclpy.init()
        self.robot = FR3Controller()
        self.camera = RS_D455(WH=[640, 480], depth_threshold=[0, 2])
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.robot)
        
        # Initialize the calibration board, this should corresponds to the real board
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard(
            (6, 9),
            squareLength=0.03,
            markerLength=0.022,
            dictionary=self.dictionary,
        )
        # Get the camera intrinsic parameters
        self.intrinsic_matrix, self.dist_coef = (
            self.camera.intrinsic_matrix,
            self.camera.dist_coef,
        )
        # Specify the poses used for the hand-eye calirbation
        # self.poses = []
        self.poses = [
            [0.6675535440444946, 0.16737790405750275, 0.21717093884944916, 2.7849929332733154, -0.12459675222635269, -0.34754031896591187],
            [0.5156394243240356, 0.1696367859840393, 0.1863502711057663, 2.744291305541992, -0.6115761995315552, 0.06089765578508377],
            [0.5777524709701538, 0.03800298646092415, 0.16350185871124268, 3.112367868423462, -0.45929813385009766, 0.024628005921840668],
            [0.5105370283126831, 0.07431235909461975, 0.1391362100839615, 2.659372568130493, -0.3047432005405426, -0.23964986205101013],
            [0.6272953152656555, -0.11205264180898666, 0.22839973866939545, 3.4260194301605225, -0.20160509645938873, 0.017706383019685745],
            [0.41143882274627686, 0.007283882703632116, 0.11021111160516739, 3.4655354022979736, -0.37139269709587097, -0.32729509472846985],
            [0.5364089608192444, 0.11824066936969757, 0.11720602214336395, 2.8426530361175537, 0.16258998215198517, -0.04695363715291023],
            [0.6777443885803223, 0.0528448186814785, 0.2255811095237732, 3.6604998111724854, -0.5709952116012573, 0.2779277265071869],
            [0.5649014115333557, -0.08028866350650787, 0.21902208030223846, 3.1961398124694824, -0.5891002416610718, -0.2354361116886139],
            [0.6881664991378784, 0.15997612476348877, 0.1765804886817932, 2.7736599445343018, 0.0948072075843811, 0.07972355931997299],
            [0.6369515061378479, -0.02644766867160797, 0.22759348154067993, 2.784287929534912, -0.16557861864566803, 0.11069400608539581],
            [0.6712604761123657, -0.06634043157100677, 0.15554828941822052, 2.750826597213745, 0.14690949022769928, 0.33572062849998474],
            [0.5799790620803833, 0.1621588170528412, 0.22765247523784637, 3.1585590839385986, -0.30459064245224, -0.013322446495294571],
            [0.5664640665054321, 0.019997483119368553, 0.12498245388269424, 2.9040133953094482, -0.2502323389053345, -0.1554417759180069],
            [0.5437729954719543, -0.024172116070985794, 0.1689884513616562, 2.8403711318969727, -0.44348785281181335, 0.031262900680303574],
            [0.5842674374580383, 0.17846082150936127, 0.20537109673023224, 3.4410297870635986, -0.3287678062915802, 0.306922048330307],
            [0.48148491978645325, 0.15327629446983337, 0.18836379051208496, 2.9361371994018555, -0.4892374873161316, 0.0371931828558445],
            [0.4270573556423187, -0.13192406296730042, 0.16012075543403625, 2.7958450317382812, 0.2777821719646454, 0.03283573314547539],
            [0.4987163245677948, -0.13186030089855194, 0.17100739479064941, 3.5709617137908936, 0.13484950363636017, 0.5037873983383179],
            [0.5838234424591064, 0.13812290132045746, 0.23884403705596924, 3.6408329010009766, 0.1482536345720291, 0.28810974955558777],
            [0.44694751501083374, -0.0353066623210907, 0.1501854509115219, 2.8190724849700928, -0.46500855684280396, 0.2745727300643921],
            [0.4849507808685303, -0.11356658488512039, 0.1250910758972168, 2.797785997390747, -0.5259429812431335, 0.41945022344589233]
            ]

        # Start executor in background
        import threading
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()


    def sample_poses(
        self,
        n: int,
        xyz_ranges: Sequence[Tuple[float, float]],
        rpy_ranges: Sequence[Tuple[float, float]],
        degrees: bool = False,
    ) -> List[List[float]]:
        """
        Sample `n` random SE(3) poses.

        Parameters
        ----------
        n : int
            Number of poses to generate.
        xyz_ranges : [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
            Position limits (meters).
        rpy_ranges : [(rmin, rmax), (pmin, pmax), (ymin, ymax)]
            Orientation limits (radians by default, or degrees if `degrees=True`).
        degrees : bool, optional
            If True, treat the rpy limits as degrees and return degrees.

        Returns
        -------
        poses : [[x, y, z, roll, pitch, yaw], ...]  (length == n)
        """
        poses = []
        for _ in range(n):
            x, y, z = (
                random.uniform(*xyz_ranges[0]),
                random.uniform(*xyz_ranges[1]),
                random.uniform(*xyz_ranges[2]),
            )
            roll, pitch, yaw = (
                random.uniform(*rpy_ranges[0]),
                random.uniform(*rpy_ranges[1]),
                random.uniform(*rpy_ranges[2]),
            )
            if degrees:
                roll, pitch, yaw = map(math.degrees, (roll, pitch, yaw))
            poses.append([x, y, z, roll, pitch, yaw])
        return poses

    def set_calibration_poses(self, poses):
        self.poses = poses


    def calibrate(
        self,
        method: str = "charuco",      # "charuco" | "apriltag"
        tag_size: float = 0.07,       # [m] AprilTag edge length
        tag_id:   int   = 1,          # AprilTag ID to track
        visualize: bool = True,
        # min_views: int = 8,           # how many stored frames before solve
        save_path: str | Path = "handeye_result.json",
    ):
        """
        Interactive data collection + hand–eye calibration.

        Keys while live window is open
        ---------------------------------
        s  – save this frame / robot pose
        q  – discard frame and sample a new pose
        ESC / window close – finish (run solve if ≥ min_views)
        """
        if method not in ("charuco", "apriltag"):
            raise ValueError("method must be 'charuco' or 'apriltag'")

        # ============ AprilTag detector & helpers =============
        fx, fy = self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]
        cx, cy = self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]
        at_det = Detector(families="tag36h11", nthreads=4,
                        quad_decimate=1.5, refine_edges=True)
        obj_pts_tag = np.array(
            [(-tag_size/2, -tag_size/2, 0), ( tag_size/2, -tag_size/2, 0),
            ( tag_size/2,  tag_size/2, 0), (-tag_size/2,  tag_size/2, 0)],
            dtype=np.float32)

        # ============ buffers =============
        R_g2b, t_g2b, R_b2c, t_b2c = [], [], [], []
        saved_poses, rgbs, depths, pts_list, masks = [], [], [], [], []

        # ============ display window =============
        win = "Live (s=save  q=next  ESC=done)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 720)

        abort = False
        while not abort:
            # -------- sample random pose & move ------------------------------
            pose = self.sample_poses(
                1,
                xyz_ranges=[(0.5, 0.7), (-0.1, 0.1), (0.2, 0.25)],
                rpy_ranges=[(7/8*math.pi, 9/8*math.pi),
                            (-1/6*math.pi, 1/6*math.pi),
                            (-1/6*math.pi, 1/6*math.pi)],
            )[0]
            print(f"\n→ Move to {pose}")
            if not self.robot.move_to(*pose):
                print("   Move failed, resampling …")
                continue
            time.sleep(2.0)  # settle

            # -------- per-pose stream loop -----------------------------------
            skip_pose = False
            while not skip_pose and not abort:
                _, rgb, depth, mask = self.camera.get_observations()
                bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                # ---------- detect board -------------------------------------
                if method == "charuco":
                    corners, ids, _ = cv2.aruco.detectMarkers(bgr, self.dictionary)
                    retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, bgr, self.board,
                        self.intrinsic_matrix, self.dist_coef)
                    ok = retval >= 6
                    if ok:
                        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            ch_corners, ch_ids, self.board,
                            self.intrinsic_matrix, self.dist_coef)
                    if ok and visualize:
                        cv2.aruco.drawDetectedCornersCharuco(bgr, ch_corners, ch_ids)

                else:  # ---------------- AprilTag --------------------------
                    dets = [d for d in at_det.detect(gray, estimate_tag_pose=False)
                            if d.tag_id == tag_id]
                    ok = len(dets) > 0
                    if ok:
                        det = dets[0]
                        pts = det.corners.astype(np.float32)
                        ok, rvec, tvec = cv2.solvePnP(
                            obj_pts_tag, pts, self.intrinsic_matrix,
                            self.dist_coef, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if visualize:
                            cv2.polylines(bgr, [pts.astype(int)], True, (0,255,0), 2)

                if ok and visualize:
                    cv2.drawFrameAxes(bgr, self.intrinsic_matrix,
                                    self.dist_coef, rvec, tvec, tag_size*0.15)

                cv2.imshow(win, bgr)
                key = cv2.waitKey(1) & 0xFF

                # ------------- key handler -----------------------------------
                if key == ord('s') and ok:
                    # store frame
                    R_b2c.append(cv2.Rodrigues(rvec)[0])
                    t_b2c.append(tvec[:, 0])

                    cur = self.robot.get_current_pose()
                    R_g2b.append(rpy_to_rotation_matrix(*cur[3:]))
                    t_g2b.append(np.array(cur[:3]))

                    saved_poses.append(pose)
                    rgbs.append(rgb); depths.append(depth)
                    pts_list.append(_); masks.append(mask)

                    print(f"   ✓ stored ({len(saved_poses)})")
                    skip_pose = True

                elif key in (ord('q'), ord('Q')):
                    skip_pose = True            # discard & sample new pose
                elif key == 27:
                    abort = True
                elif key == -1 and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    abort = True

            # # stop automatically once we have enough views
            # if len(saved_poses) >= min_views and not abort:
            #     print(f"\nCollected {len(saved_poses)} views  →  solving …")
            #     break

        cv2.destroyWindow(win)

        # if len(saved_poses) < min_views:
        #     print("Not enough data collected; calibration aborted.")
        #     return

        # ================= hand–eye solve ==========================
        R_b2g, t_b2g = [], []
        for R, t in zip(R_g2b, t_g2b):
            R_b2g.append(R.T)
            t_b2g.append(-R.T @ t)

        R_b2board, t_b2board, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
            R_world2cam = R_b2c,  
            t_world2cam = t_b2c,
            R_base2gripper = R_b2g, 
            t_base2gripper = t_b2g,
            R_base2world=None,
            t_base2world=None,
            R_gripper2cam=None,
            t_gripper2cam=None,
            method = cv2.CALIB_HAND_EYE_TSAI)

        result = dict(
            method           = method,
            tag_size_m       = tag_size,
            tag_id           = tag_id,
            R_cam2gripper    = R_g2c.T.tolist(),
            t_cam2gripper    = (-R_g2c.T @ t_g2c[:, 0]).tolist(),
            R_board2base     = R_b2board.T.tolist(),
            t_board2base     = (-R_b2board.T @ t_b2board[:, 0]).tolist(),
            saved_poses      = saved_poses,
            n_views          = len(saved_poses),
        )

        # ============== save =======================================
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nCalibration written to {save_path}")
        print("R_cam2gripper:\n", np.array(result["R_cam2gripper"]))
        print("t_cam2gripper:\n", np.array(result["t_cam2gripper"]))


    # def calibrate(
    #     self,
    #     method: str = "charuco",          # "charuco" | "apriltag"
    #     tag_size: float = 0.07,           # [m] edge length if using AprilTag
    #     tag_id:   int   = 1,              # which tag to use (AprilTag mode)
    #     visualize: bool = True,
    # ):
    #     """Hand–eye calibration with either a Charuco board or a single AprilTag."""
    #     if method not in ("charuco", "apriltag"):
    #         raise ValueError("method must be 'charuco' or 'apriltag'")

    #     # ------------------------------------------------------------------ collect
    #     R_g2b, t_g2b, R_b2c, t_b2c = [], [], [], []
    #     rgbs, depths, pts_list, masks = [], [], [], []

    #     if method == "apriltag":
    #         fx, fy = self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]
    #         cx, cy = self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]
    #         at_det = Detector(families="tag36h11", nthreads=4,
    #                         quad_decimate=1.5, refine_edges=True)

    #         obj_pts = np.array([                 # tag corners in its own frame
    #             [-tag_size/2, -tag_size/2, 0],
    #             [ tag_size/2, -tag_size/2, 0],
    #             [ tag_size/2,  tag_size/2, 0],
    #             [-tag_size/2,  tag_size/2, 0]], dtype=np.float32)

    #     for pose in self.poses:
            
    #         self.robot.move_to(*pose)
    #         time.sleep(5)

    #         pts, rgb, depth, mask = self.camera.get_observations()
    #         img = (rgb * 255).astype(np.uint8)
    #         img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #         # ---------------------------------------------------------------- detect
    #         if method == "charuco":
    #             corners, ids, _ = cv2.aruco.detectMarkers(
    #                 img_bgr, self.dictionary)
    #             retval, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
    #                 corners, ids, img_bgr, self.board,
    #                 self.intrinsic_matrix, self.dist_coef)

    #             if retval < 6:           # need enough corners
    #                 print("❌ Not enough Charuco corners – skipping frame")
    #                 continue

    #             ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
    #                 char_corners, char_ids, self.board,
    #                 self.intrinsic_matrix, self.dist_coef)
    #             if not ok:
    #                 print("❌ Charuco pose fail – skipping")
    #                 continue

    #             if visualize:
    #                 cv2.aruco.drawDetectedCornersCharuco(
    #                     img_bgr, char_corners, char_ids)
    #                 cv2.imshow("charuco", img_bgr); cv2.waitKey(10)

    #         else:   # ---------------------- AprilTag -----------------------
    #             gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #             dets = [d for d in at_det.detect(
    #                 gray, estimate_tag_pose=False)
    #                     if d.tag_id == tag_id]
    #             if not dets:
    #                 print(f"❌ Tag {tag_id} not found – skipping")
    #                 continue

    #             det = dets[0]
    #             corners = det.corners.astype(np.float32)
    #             # use solvePnP for consistency (Pose from detector is OK too)
    #             ok, rvec, tvec = cv2.solvePnP(
    #                 obj_pts, corners, self.intrinsic_matrix,
    #                 self.dist_coef, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    #             if not ok:
    #                 print("❌ PnP failed – skipping")
    #                 continue

    #             if visualize:
    #                 cv2.polylines(img_bgr, [corners.astype(int)], True, (0,255,0), 2)
    #                 cv2.drawFrameAxes(img_bgr, self.intrinsic_matrix,
    #                                 self.dist_coef, rvec, tvec, tag_size*0.5)
    #                 cv2.imshow("apriltag", img_bgr); cv2.waitKey(10)

    #         # ---------------------------------------------------------------- store
    #         R_b2c.append(cv2.Rodrigues(rvec)[0])
    #         t_b2c.append(tvec[:, 0])

    #         cur = self.robot.get_current_pose()          # TCP pose list
    #         R_g2b.append(rpy_to_rotation_matrix(*cur[3:]))
    #         t_g2b.append(np.array(cur[:3]) / 1000.0)

    #         rgbs.append(rgb); depths.append(depth)
    #         pts_list.append(pts); masks.append(mask)

    #     if len(R_b2c) < 5:
    #         raise RuntimeError("Not enough valid views collected for calibration.")

    #     # ------------------------------------------------------------------ solve
    #     R_b2g, t_b2g = [], []
    #     for R, t in zip(R_g2b, t_g2b):
    #         R_b2g.append(R.T);                t_b2g.append(-R.T @ t)

    #     R_b2board, t_b2board, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
    #         R_world2cam = R_b2c,  t_world2cam = t_b2c,
    #         R_base2gripper = R_b2g, t_base2gripper = t_b2g,
    #         method = cv2.CALIB_HAND_EYE_TSAI)

    #     # ------------------------------------------------------------------ save
    #     res = dict(
    #         R_cam2gripper = R_g2c.T,
    #         t_cam2gripper = -R_g2c.T @ t_g2c[:,0],
    #         R_board2base  = R_b2board.T,
    #         t_board2base  = -R_b2board.T @ t_b2board[:,0],
    #         R_gripper2base= R_g2b,  t_gripper2base=t_g2b,
    #         R_board2cam   = R_b2c,  t_board2cam  = t_b2c,
    #         rgbs=rgbs, depths=depths, point_list=pts_list, masks=masks,
    #         poses=self.poses,  method=method)

    #     self._save_results(res, f"calibrate_{method}.pkl")
    #     print("\n=== Calibration result ===")
    #     print("R_cam2gripper:\n", res["R_cam2gripper"])
    #     print("t_cam2gripper:\n", res["t_cam2gripper"])


    # def calibrate(self, visualize=True):
    #     R_gripper2base = []
    #     t_gripper2base = []
    #     R_board2cam = []
    #     t_board2cam = []
    #     rgbs = []
    #     depths = []
    #     point_list = []
    #     masks = []

    #     for pose in self.poses:
    #         # Move to the pose and wait for 5s to make it stable
    #         self.robot.move_to_pose(pose=pose, wait=True, ignore_error=True) # TODO: move_to
    #         time.sleep(5)

    #         # Calculate the markers
    #         points, colors, depth_img, mask = self.camera.get_observations()
    #         calibration_img = colors.copy()
    #         calibration_img *= 255
    #         calibration_img = calibration_img.astype(np.uint8)
    #         calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_RGB2BGR)
    #         # calibration_img, depth_img = self.camera.get_observations(only_raw=True)

    #         corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
    #             image=calibration_img,
    #             dictionary=self.dictionary,
    #             parameters=None,
    #         )

    #         # Calculate the charuco corners
    #         retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
    #             markerCorners=corners,
    #             markerIds=ids,
    #             image=calibration_img,
    #             board=self.board,
    #             cameraMatrix=self.intrinsic_matrix,
    #             distCoeffs=self.dist_coef,
    #         )

    #         print("number of corners: ", len(charuco_corners))

    #         if visualize:
    #             cv2.aruco.drawDetectedCornersCharuco(
    #                 image=calibration_img,
    #                 charucoCorners=charuco_corners,
    #                 charucoIds=charuco_ids,
    #             )
    #             cv2.imshow("cablibration", calibration_img)
    #             cv2.waitKey(1)

    #         rvec = None
    #         tvec = None
    #         retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
    #             charuco_corners,
    #             charuco_ids,
    #             self.board,
    #             self.intrinsic_matrix,
    #             self.dist_coef,
    #             rvec,
    #             tvec,
    #         )
    #         if not retval:
    #             raise ValueError("pose estimation failed")

    #         reprojected_points, _ = cv2.projectPoints(
    #             self.board.getChessboardCorners()[charuco_ids, :],
    #             rvec,
    #             tvec,
    #             self.intrinsic_matrix,
    #             self.dist_coef,
    #         )

    #         # Reshape for easier handling
    #         reprojected_points = reprojected_points.reshape(-1, 2)
    #         charuco_corners = charuco_corners.reshape(-1, 2)

    #         # Calculate the error
    #         error = np.sqrt(
    #             np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
    #         ).mean()

    #         print("Reprojection Error:", error)

    #         # if error < 0.3:
    #         print("Pose estimation succeed!")
    #         # Save the transformation of board2cam
    #         R_board2cam.append(cv2.Rodrigues(rvec)[0])
    #         t_board2cam.append(tvec[:, 0])
    #         # Save the transformation of the gripper2base
    #         current_pose = self.robot.get_current_pose()
    #         print("Current pose: ", current_pose)
    #         R_gripper2base.append(
    #             rpy_to_rotation_matrix(
    #                 current_pose[3], current_pose[4], current_pose[5]
    #             )
    #         )
    #         t_gripper2base.append(np.array(current_pose[:3]) / 1000)
    #         # Save the rgb and depth images
    #         rgbs.append(colors)
    #         depths.append(depth_img)
    #         point_list.append(points)
    #         masks.append(mask)

    #     R_base2gripper = []
    #     t_base2gripper = []
    #     for i in range(len(R_gripper2base)):
    #         R_base2gripper.append(R_gripper2base[i].T)
    #         t_base2gripper.append(-R_gripper2base[i].T @ t_gripper2base[i])

    #     # Do the robot-world hand-eye calibration
    #     (
    #         R_base2board,
    #         t_base2board,
    #         R_gripper2cam,
    #         t_gripper2cam,
    #     ) = cv2.calibrateRobotWorldHandEye(
    #         R_world2cam=R_board2cam,
    #         t_world2cam=t_board2cam,
    #         R_base2gripper=R_base2gripper,
    #         t_base2gripper=t_base2gripper,
    #         R_base2world=None,
    #         t_base2world=None,
    #         R_gripper2cam=None,
    #         t_gripper2cam=None,
    #         method=cv2.CALIB_HAND_EYE_TSAI,
    #     )

    #     R_cam2gripper = R_gripper2cam.T
    #     t_cam2gripper = -R_gripper2cam.T @ t_gripper2cam[:, 0]

    #     R_board2base = R_base2board.T
    #     t_board2base = -R_base2board.T @ t_base2board[:, 0]

    #     results = {}
    #     results["R_cam2gripper"] = R_cam2gripper
    #     results["t_cam2gripper"] = t_cam2gripper
    #     results["R_board2base"] = R_board2base
    #     results["t_board2base"] = t_board2base
    #     results["R_gripper2base"] = R_gripper2base
    #     results["t_gripper2base"] = t_gripper2base
    #     results["R_board2cam"] = R_board2cam
    #     results["t_board2cam"] = t_board2cam
    #     results["rgbs"] = rgbs
    #     results["depths"] = depths
    #     results["point_list"] = point_list
    #     results["masks"] = masks
    #     results["poses"] = self.poses

    #     self._save_results(results, "calibrate.pkl")

    #     print(R_cam2gripper)
    #     print(t_cam2gripper)



    def test_pose(self):
        """Move to each pose and continuously track an AprilTag in the live stream.

        1. Press 'q' to move on to the next robot pose without saving the pose, 
        2. Press 's' to save the pose and move to the next pose
        3. Press ESC/Window close to abort.
        """
        # ----- 1. choose AprilTag detector backend ----------------------------

        fx, fy = self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]
        cx, cy = self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]
        at_detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.5,       # speed-accuracy trade-off
            refine_edges=True,
            decode_sharpening=0.25,
        )


        tag_size = 0.07  # metres  (edge length of your tag)
        K = self.intrinsic_matrix
        D = self.dist_coef

        win = "AprilTag Tracking  (s: save  q: next  ESC: quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)

        saved_poses = []           # list of [x,y,z,r,p,y]
        user_abort  = False

        while not user_abort:

            pose = self.sample_poses(
                n=1,
                xyz_ranges=[(0.45, 0.65), (-0.15, 0.15), (0.15, 0.2)],
                rpy_ranges=[(5/6*math.pi, 7/6*math.pi),
                            (-1/5*math.pi, 1/5*math.pi),
                            (-1/6*math.pi, 1/6*math.pi)],
            )[0]
            print(f"\n=== Pose {pose} ===")
            
            if not self.robot.move_to(*pose):
                print("   ❌ Move failed, sampling a new pose …")
                continue
            time.sleep(2.0)   

            next_pose = False 
            while not next_pose and not user_abort:
                _, rgb, _, _ = self.camera.get_observations()
                frame = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                dets = at_detector.detect(
                    gray, estimate_tag_pose=True,
                    camera_params=(fx, fy, cx, cy), tag_size=tag_size)

                for det in dets:
                    pts = det.corners.astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cv2.putText(frame, f"id:{det.tag_id}", tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.drawFrameAxes(frame, K, D, det.pose_R, det.pose_t, tag_size*0.3)

                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF

                # ---------- key handling -------------------------------------
                if k == ord('s'):                    # save current robot pose
                    saved_poses.append(pose)
                    print(f'saved pose: {pose}.')
                    print(f"   ✓ Pose saved ({len(saved_poses)} total)")
                    next_pose = True

                elif k in (ord('q'), ord('Q')):      # next random pose
                    next_pose = True
                elif k == 27:                        # ESC
                    user_abort = True

                # window closed?
                if k == -1 and cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    user_abort = True
        # ---------------------------------------------------------------------
        cv2.destroyWindow(win)

        # 4. write & report
        if saved_poses:
            np.save("saved_poses.npy", np.array(saved_poses, dtype=np.float32))
            print("\nSaved poses array (also written to saved_poses.npy):")
            print(np.array(saved_poses, dtype=np.float32))
        else:
            print("\nNo poses were saved.")




    def _save_results(self, results, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(results, f)


    def close(self):
        try:
            self.robot.destroy_node()
        except:
            pass
        self._executor.shutdown()
        rclpy.shutdown()




if __name__ == "__main__":
    calibrate_env = RoboCalibrate()

    # Test and save candidate calibration poses
    # calibrate_env.test_pose()
    # # Calibrate the camera using collected poses
    # calibrate_env.calibrate(method="apriltag", tag_id=5)
    calibrate_env.calibrate() 

    calibrate_env.close()






    # poses = calibrate_env.sample_poses(
    #             n=18,
    #             xyz_ranges=[(0.4, 0.7), (-0.18, 0.18), (0.10, 0.25)],
    #             rpy_ranges=[(5/6*math.pi, 7/6*math.pi),
    #                         (-1/5*math.pi, 1/5*math.pi),
    #                         (-1/6*math.pi, 1/6*math.pi)],
    #         )
    
    # print('sample poses: ')
    # for p in poses:
    #     print(p)
    
    # calibrate_env.set_calibration_poses(poses=poses)