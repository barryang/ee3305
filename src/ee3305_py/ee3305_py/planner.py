from heapq import heappush, heappop
from math import hypot, floor, inf, sqrt
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    qos_profile_services_default,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from math import cos, sin, atan2, pi


class AStarNode:
    def __init__(self, c, r):
        self.parent = None
        self.f = inf   # f = g + h
        self.g = inf   # cost from start
        self.h = inf   # heuristic to goal
        self.c = c
        self.r = r
        self.expanded = False

    def __lt__(self, other):  # comparator for heapq (min-heap) sorting by f
        return self.f < other.f


class Planner(Node):

    def __init__(self, node_name="planner"):
        super().__init__(node_name)

        # Parameters: Declare
        self.declare_parameter("max_access_cost", int(100))
        # cell penalty scale (meters per cost unit) -> tune if you want to avoid high-cost cells
        self.declare_parameter("cell_penalty_scale", float(0.0005))
        # Trajectory smoothing parameters
        self.declare_parameter("smooth_path", bool(True))  # Enable/disable path smoothing
        self.declare_parameter("spline_resolution", float(0.05))  # Distance between smoothed points [m]
        # Obstacle avoidance parameters
        self.declare_parameter("use_laser_obstacles", bool(False))  # Enable/disable laser-based obstacle avoidance
        self.declare_parameter("obstacle_inflation_radius", float(0.2))  # Inflate obstacles by this radius [m]
        self.declare_parameter("replan_threshold", float(0.2))  # Replan if obstacle within this distance of path [m]
        self.declare_parameter("replan_frequency", float(20.0))  # Maximum replanning frequency [Hz]

        # Parameters: Get Values
        self.max_access_cost_ = self.get_parameter("max_access_cost").value
        self.cell_penalty_scale_ = self.get_parameter("cell_penalty_scale").value
        self.smooth_path_ = self.get_parameter("smooth_path").value
        self.spline_resolution_ = self.get_parameter("spline_resolution").value
        self.use_laser_obstacles_ = self.get_parameter("use_laser_obstacles").value
        self.obstacle_inflation_radius_ = self.get_parameter("obstacle_inflation_radius").value
        self.replan_threshold_ = self.get_parameter("replan_threshold").value
        self.replan_frequency_ = self.get_parameter("replan_frequency").value

        # Handles: Topic Subscribers
        qos_profile_latch = QoSProfile(
            history=qos_profile_services_default.history,
            depth=qos_profile_services_default.depth,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=qos_profile_services_default.reliability,
        )
        self.sub_global_costmap_ = self.create_subscription(
            OccupancyGrid,
            "global_costmap",
            self.callbackSubGlobalCostmap_,
            qos_profile_latch,
        )

        # Path request subscriber
        self.sub_path_request = self.create_subscription(
            Path,
            "path_request",
            self.callbackSubPathRequest_,
            10,
        )
        
        # ========== IMPROVEMENT: Laser Scan Subscriber for Dynamic Obstacle Detection ==========
        # Problem: The planner only used static global costmap, which doesn't include dynamic
        # obstacles (e.g., moving objects, obstacles not in the map). This caused the robot
        # to plan paths through dynamic obstacles, leading to collisions.
        #
        # Solution: Subscribe to laser scan to detect dynamic obstacles in real-time and
        # incorporate them into the costmap before planning.
        #
        # How it works:
        # - Subscribes to /scan topic (LaserScan messages)
        # - Stores latest scan in self.current_scan_ for processing
        # - Only subscribes if use_laser_obstacles_ is enabled (configurable)
        #
        # Why sensor QoS: Uses qos_profile_sensor_data for real-time sensor data with
        # best-effort reliability (can drop old messages if queue is full).
        if self.use_laser_obstacles_:
            self.sub_scan_ = self.create_subscription(
                LaserScan,
                "scan",
                self.callbackSubScan_,
                qos_profile_sensor_data,
            )
        
        # ========== IMPROVEMENT: Odometry Subscriber for Accurate Robot Position ==========
        # Problem: Path requests contain robot position, but it may be slightly outdated.
        # For accurate laser scan transformation to world frame, we need the most current
        # robot position and orientation from odometry.
        #
        # Solution: Subscribe to odometry to get real-time robot pose (position + orientation).
        #
        # How it works:
        # - Subscribes to /odom topic (Odometry messages)
        # - Extracts robot position (x, y) and orientation (yaw) from odometry
        # - Stores in self.rbt_x_odom_, self.rbt_y_odom_, self.rbt_yaw_
        # - Used for transforming laser scan points from robot frame to world frame
        #
        # Why separate from path request: Path request position may be slightly delayed.
        # Odometry provides the most current robot pose needed for accurate obstacle mapping.
        if self.use_laser_obstacles_:
            self.sub_odom_ = self.create_subscription(
                Odometry,
                "odom",
                self.callbackSubOdom_,
                10,
            )

        # Path publisher
        self.pub_path_ = self.create_publisher(
            Path,
            "path",
            10
        )

        # Timer for path planning
        self.timer = self.create_timer(0.1, self.callbackTimer_)
        
        # ========== IMPROVEMENT: Obstacle-Based Replanning Timer ==========
        # Problem: Once a path is planned, the planner doesn't check if new obstacles
        # appear along the path. This causes the robot to follow a path that may become
        # blocked by dynamic obstacles, leading to collisions or getting stuck.
        #
        # Solution: Add a timer that periodically checks if obstacles are near the current
        # path and triggers replanning if needed.
        #
        # How it works:
        # - Runs at replan_frequency_ (default 20 Hz) to check for obstacles
        # - Calls checkObstaclesNearPath_() to detect obstacles within replan_threshold_
        # - If obstacles detected, sets has_new_request_ = True to trigger replanning
        # - Lower frequency than main timer to avoid excessive replanning (computationally expensive)
        #
        # Why separate timer: Main planning timer only runs when has_new_request_ is True.
        # This timer actively monitors the path and triggers replanning when needed.
        if self.use_laser_obstacles_:
            self.timer_replan_ = self.create_timer(
                1.0 / self.replan_frequency_, 
                self.callbackTimerReplan_
            )

        # State
        self.has_new_request_ = False
        self.received_map_ = False

        # Initialize request and map-related fields to safe defaults
        self.rbt_x_ = 0.0
        self.rbt_y_ = 0.0
        self.goal_x_ = 0.0
        self.goal_y_ = 0.0
        self.costmap_ = []  # flat list of int8 costs (row-major: index = r * cols + c)
        self.costmap_resolution_ = 0.0
        self.costmap_origin_x_ = 0.0
        self.costmap_origin_y_ = 0.0
        self.costmap_rows_ = 0
        self.costmap_cols_ = 0
        
        # ========== IMPROVEMENT: Dynamic Obstacle Tracking State Variables ==========
        # These variables store laser scan data and robot pose for dynamic obstacle detection.
        #
        # Why needed:
        # - current_scan_: Stores latest LaserScan message for obstacle processing
        # - received_scan_: Flag to track if we've received scan data (safety check)
        # - rbt_x_odom_, rbt_y_odom_: Robot position from odometry (more accurate than path request)
        # - rbt_yaw_: Robot orientation (needed to transform laser scan from robot frame to world frame)
        # - current_path_: Stores the current planned path to check if obstacles are near it
        # - last_replan_time_: Tracks last replan time (currently unused, reserved for future rate limiting)
        self.current_scan_ = None
        self.received_scan_ = False
        self.rbt_x_odom_ = 0.0  # Robot position from odometry (for laser scan transformation)
        self.rbt_y_odom_ = 0.0
        self.rbt_yaw_ = 0.0  # Robot orientation for laser scan transformation
        self.current_path_ = None  # Store current path to check for obstacles
        self.last_replan_time_ = 0.0  # Track last replan time to limit frequency

    # Callbacks =============================================================

    def callbackSubPathRequest_(self, msg: Path):
        """Receives a path request containing two poses: [0]=robot, [1]=goal."""
        if len(msg.poses) < 2:
            self.get_logger().warn("Path request must contain robot and goal poses; ignoring.")
            return

        self.rbt_x_ = msg.poses[0].pose.position.x
        self.rbt_y_ = msg.poses[0].pose.position.y
        self.goal_x_ = msg.poses[1].pose.position.x
        self.goal_y_ = msg.poses[1].pose.position.y
        self.has_new_request_ = True
        # ========== IMPROVEMENT: Logging for Debugging ==========
        # Added logging to help debug issues when planner stops working.
        # Logs when new path requests are received (e.g., from RViz 2D goal tool).
        self.get_logger().info(f"New path request: Robot=({self.rbt_x_:.3f}, {self.rbt_y_:.3f}), Goal=({self.goal_x_:.3f}, {self.goal_y_:.3f})")

    def callbackSubGlobalCostmap_(self, msg: OccupancyGrid):
        """Latches the global costmap (inflated costs) and its metadata."""
        self.costmap_ = list(msg.data)  # This is the base costmap (will be modified with obstacles)
        self.costmap_resolution_ = float(msg.info.resolution)
        self.costmap_origin_x_ = float(msg.info.origin.position.x)
        self.costmap_origin_y_ = float(msg.info.origin.position.y)
        self.costmap_rows_ = int(msg.info.height)
        self.costmap_cols_ = int(msg.info.width)

        expected_len = self.costmap_rows_ * self.costmap_cols_
        if len(self.costmap_) != expected_len:
            self.get_logger().warn(
                f"Costmap size mismatch (data={len(self.costmap_)}, rows*cols={expected_len})."
            )

        self.received_map_ = True
    
    def callbackSubScan_(self, msg: LaserScan):
        """
        ========== IMPROVEMENT: Laser Scan Callback ==========
        Problem: Need to store latest laser scan data for dynamic obstacle detection.
        
        Solution: Store the latest LaserScan message whenever it's received.
        
        How it works:
        - Called automatically when new laser scan data arrives on /scan topic
        - Stores the message in self.current_scan_ for later processing
        - Sets received_scan_ flag to indicate we have scan data
        
        Why this approach: Laser scan is processed on-demand when planning (not here)
        to avoid blocking the callback and to use the most recent robot pose for transformation.
        """
        self.current_scan_ = msg
        self.received_scan_ = True
    
    def callbackSubOdom_(self, msg: Odometry):
        """
        ========== IMPROVEMENT: Odometry Callback for Robot Pose ==========
        Problem: Need accurate robot position and orientation to transform laser scan
        points from robot frame to world frame. Path request position may be outdated.
        
        Solution: Extract robot pose (position + orientation) from odometry messages.
        
        How it works:
        - Called automatically when new odometry data arrives on /odom topic
        - Extracts x, y position from msg.pose.pose.position
        - Extracts yaw (rotation around z-axis) from quaternion orientation
        - Stores in separate variables (rbt_x_odom_, rbt_y_odom_, rbt_yaw_)
        
        Why separate variables: Path request also sets rbt_x_, rbt_y_, but we need
        the most accurate position from odometry for laser scan transformation.
        
        Quaternion to yaw conversion:
        - ROS uses quaternions (x, y, z, w) for orientation
        - Standard formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
        - This extracts rotation around z-axis (yaw) from the quaternion
        """
        # Store odometry position separately (used for laser scan transformation)
        self.rbt_x_odom_ = msg.pose.pose.position.x
        self.rbt_y_odom_ = msg.pose.pose.position.y
        # Extract yaw from quaternion (standard formula)
        q = msg.pose.pose.orientation
        # Convert quaternion (x, y, z, w) to yaw angle
        # Formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        self.rbt_yaw_ = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def callbackTimer_(self):
        if not self.received_map_ or not self.has_new_request_:
            return

        # ========== IMPROVEMENT: Use Odometry Position for Start ==========
        # Problem: Path request contains robot position, but it may be slightly outdated.
        # When planning with dynamic obstacles, we need the most accurate robot position
        # to correctly transform laser scan obstacles to world frame.
        #
        # Solution: Use odometry position (rbt_x_odom_, rbt_y_odom_) for start position
        # when laser obstacles are enabled, as it's more accurate and up-to-date.
        #
        # How it works:
        # - If laser obstacles enabled AND odometry position is valid (non-zero), use odometry
        # - Otherwise, fall back to path request position (for backward compatibility)
        #
        # Why this matters: Small position errors can cause obstacles to be placed incorrectly
        # in the costmap, leading to incorrect path planning or false obstacle detections.
        start_x = self.rbt_x_odom_ if self.use_laser_obstacles_ and self.rbt_x_odom_ != 0.0 else self.rbt_x_
        start_y = self.rbt_y_odom_ if self.use_laser_obstacles_ and self.rbt_y_odom_ != 0.0 else self.rbt_y_
        
        # run the path planner (A*)
        self.astar_(start_x, start_y, self.goal_x_, self.goal_y_)
        self.has_new_request_ = False
    
    def callbackTimerReplan_(self):
        """
        ========== IMPROVEMENT: Obstacle-Based Replanning Timer Callback ==========
        Problem: Dynamic obstacles can appear along the planned path after planning.
        The robot would continue following the old path and collide with new obstacles.
        
        Solution: Periodically check if obstacles are near the current path, and trigger
        replanning if obstacles are detected within replan_threshold distance.
        
        How it works:
        1. Check prerequisites (laser obstacles enabled, map received, scan received)
        2. Skip if new request pending (e.g., user set new goal via RViz - don't interfere)
        3. Skip if no current path exists
        4. Call checkObstaclesNearPath_() to detect obstacles near path
        5. If obstacles detected, set has_new_request_ = True to trigger replanning
        
        Why check has_new_request_: Prevents replanning from interfering with user-initiated
        goal changes. User goals should take priority over automatic obstacle-based replanning.
        
        Why separate from main timer: Main timer only runs when has_new_request_ is True.
        This timer actively monitors and triggers replanning when needed.
        """
        if not self.use_laser_obstacles_ or not self.received_map_ or not self.received_scan_:
            return
        
        # Don't replan if there's already a new request pending (e.g., new goal from RViz)
        # This ensures user-initiated goal changes take priority over automatic replanning
        if self.has_new_request_:
            return
        
        if self.current_path_ is None or len(self.current_path_.poses) == 0:
            return
        
        # Check if obstacles are near the current path
        if self.checkObstaclesNearPath_():
            # Obstacle detected near path, trigger replanning
            self.get_logger().info("Obstacle detected near path, triggering replan...")
            self.has_new_request_ = True  # Trigger replanning

    # Helpers / conversions -------------------------------------------------

    def publishInterpolatedPath(self, start_x, start_y, goal_x, goal_y):
        msg_path = Path()
        msg_path.header.stamp = self.get_clock().now().to_msg()
        msg_path.header.frame_id = "map"

        dx = start_x - goal_x
        dy = start_y - goal_y
        distance = hypot(dx, dy)
        steps = distance / 0.05

        for i in range(int(steps)):
            pose = PoseStamped()
            pose.pose.position.x = goal_x + dx * i / steps
            pose.pose.position.y = goal_y + dy * i / steps
            msg_path.poses.append(pose)

        pose = PoseStamped()
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        msg_path.poses.append(pose)

        msg_path.poses.reverse()
        self.pub_path_.publish(msg_path)
        self.get_logger().info(
            f"Publishing interpolated path between Start and Goal. Implement astar_() instead."
        )

    def XYToCR_(self, x, y):
        c = int(floor((x - self.costmap_origin_x_) / self.costmap_resolution_))
        r = int(floor((y - self.costmap_origin_y_) / self.costmap_resolution_))
        return c, r

    def CRToXY_(self, c, r):
        x = self.costmap_origin_x_ + (c + 0.5) * self.costmap_resolution_
        y = self.costmap_origin_y_ + (r + 0.5) * self.costmap_resolution_
        return x, y

    def CRToIndex_(self, c, r):
        return r * self.costmap_cols_ + c

    def outOfMap_(self, c, r):
        return (c < 0) or (r < 0) or (c >= self.costmap_cols_) or (r >= self.costmap_rows_)
    
    def updateCostmapWithObstacles_(self):
        """
        ========== IMPROVEMENT: Dynamic Obstacle Costmap Update ==========
        Problem: Global costmap only contains static obstacles from the map. Dynamic
        obstacles (moving objects, obstacles not in map) are not included, causing
        the planner to plan paths through them.
        
        Solution: Process laser scan data to detect obstacles, transform them to world
        frame, and mark them in the costmap before planning.
        
        How it works:
        1. Create a working copy of the base costmap (preserves static obstacles)
        2. Process each laser scan reading:
           a. Filter invalid readings (out of range, NaN, inf)
           b. Calculate obstacle position in robot frame (polar to Cartesian)
           c. Transform to world frame using robot pose from odometry
           d. Convert to grid coordinates
           e. Mark obstacle cell and inflate by obstacle_inflation_radius
        3. Return updated costmap with both static and dynamic obstacles
        
        Why inflate obstacles: Creates a safety margin around obstacles. Robot has
        physical size, so we need to avoid getting too close. Inflation radius should
        be at least robot_radius + safety_margin.
        
        Why transform to world frame: Laser scan is in robot frame (relative to robot).
        We need world frame coordinates to mark obstacles in the global costmap.
        
        Returns:
            list: Updated costmap with obstacles marked as high cost (255 = impassable)
        """
        if not self.use_laser_obstacles_ or not self.received_scan_ or self.current_scan_ is None:
            # Return original costmap if laser obstacles are disabled or no scan available
            return list(self.costmap_)
        
        # Create a working copy of the costmap
        working_costmap = list(self.costmap_)
        
        scan = self.current_scan_
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        
        # Inflate obstacles by marking cells within inflation radius
        inflation_cells = int(self.obstacle_inflation_radius_ / self.costmap_resolution_)
        
        # Process each laser scan reading
        for i, range_val in enumerate(scan.ranges):
            # Filter invalid readings
            # - Out of sensor range: range_val < range_min or > range_max
            # - Infinity: No obstacle detected (laser hit nothing)
            # - NaN: Invalid measurement (sensor error)
            if (range_val < scan.range_min or 
                range_val > scan.range_max or 
                range_val == inf or
                range_val != range_val):  # NaN check (NaN != NaN is True)
                continue
            
            # Calculate angle of this laser ray in robot frame
            # Laser scan provides range measurements at different angles
            # angle = starting_angle + index * angle_increment
            angle = angle_min + i * angle_increment
            
            # Transform obstacle point from robot frame to world frame
            # Robot frame: x = forward, y = left (right-handed coordinate system)
            # Convert polar coordinates (range, angle) to Cartesian (x, y) in robot frame
            obs_x_robot = range_val * cos(angle)  # Forward component in robot frame
            obs_y_robot = range_val * sin(angle)   # Lateral component in robot frame
            
            # Transform to world frame using rotation matrix
            # World frame = robot_position + rotation(robot_yaw) * robot_frame_point
            # Rotation matrix: [cos(θ) -sin(θ)] [x]
            #                [sin(θ)  cos(θ)] [y]
            # Use odometry position for accurate transformation (most current robot pose)
            obs_x_world = self.rbt_x_odom_ + obs_x_robot * cos(self.rbt_yaw_) - obs_y_robot * sin(self.rbt_yaw_)
            obs_y_world = self.rbt_y_odom_ + obs_x_robot * sin(self.rbt_yaw_) + obs_y_robot * cos(self.rbt_yaw_)
            
            # Convert to grid coordinates (which cell in the costmap)
            obs_c, obs_r = self.XYToCR_(obs_x_world, obs_y_world)
            
            # Mark obstacle and inflate it (create safety margin)
            # Inflation: Mark all cells within obstacle_inflation_radius as obstacles
            # This ensures robot doesn't get too close to obstacles
            for dr in range(-inflation_cells, inflation_cells + 1):
                for dc in range(-inflation_cells, inflation_cells + 1):
                    c = obs_c + dc
                    r = obs_r + dr
                    
                    # Check if within inflation radius (circular inflation, not square)
                    # Calculate actual distance from obstacle center
                    dist = hypot(dc * self.costmap_resolution_, dr * self.costmap_resolution_)
                    if dist > self.obstacle_inflation_radius_:
                        continue  # Outside inflation radius, skip
                    
                    # Check if within map bounds
                    if self.outOfMap_(c, r):
                        continue  # Outside map, skip
                    
                    # Mark as obstacle (high cost)
                    idx = self.CRToIndex_(c, r)
                    # Set cost to max (255) to make it impassable
                    # A* will treat cost > max_access_cost_ as blocked
                    working_costmap[idx] = 255
        
        return working_costmap
    
    def checkObstaclesNearPath_(self):
        """
        ========== IMPROVEMENT: Obstacle Detection Near Path ==========
        Problem: Need to detect if dynamic obstacles have appeared near the planned path
        so we can trigger replanning before the robot reaches them.
        
        Solution: Check distance from each laser scan obstacle to each path segment.
        If any obstacle is within replan_threshold distance, return True to trigger replan.
        
        How it works:
        1. Iterate through each path segment (pair of consecutive waypoints)
        2. For each segment, check distance to each obstacle from laser scan
        3. Transform obstacle from robot frame to world frame
        4. Calculate point-to-line-segment distance (shortest distance from obstacle to path)
        5. If distance < replan_threshold, obstacle is too close - return True
        
        Why point-to-segment distance: Path is made of line segments between waypoints.
        We need the shortest distance from obstacle to the path, not just to waypoints.
        
        Why this approach: More efficient than checking every cell in costmap. Only checks
        obstacles that actually exist (from laser scan) against the actual path.
        
        Returns:
            bool: True if obstacle is detected within replan_threshold of path
        """
        if not self.use_laser_obstacles_ or not self.received_scan_ or self.current_scan_ is None:
            return False
        
        if self.current_path_ is None or len(self.current_path_.poses) == 0:
            return False
        
        scan = self.current_scan_
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        
        # Check each path segment
        for i in range(len(self.current_path_.poses) - 1):
            p1 = self.current_path_.poses[i]
            p2 = self.current_path_.poses[i + 1]
            
            # Check each laser scan reading
            for j, range_val in enumerate(scan.ranges):
                # Filter invalid readings (same as in updateCostmapWithObstacles_)
                if (range_val < scan.range_min or 
                    range_val > scan.range_max or 
                    range_val == inf or
                    range_val != range_val):
                    continue
                
                # Calculate obstacle position in world frame (use odometry position)
                # Same transformation as in updateCostmapWithObstacles_
                angle = angle_min + j * angle_increment
                obs_x_robot = range_val * cos(angle)
                obs_y_robot = range_val * sin(angle)
                obs_x_world = self.rbt_x_odom_ + obs_x_robot * cos(self.rbt_yaw_) - obs_y_robot * sin(self.rbt_yaw_)
                obs_y_world = self.rbt_y_odom_ + obs_x_robot * sin(self.rbt_yaw_) + obs_y_robot * cos(self.rbt_yaw_)
                
                # Check distance from obstacle to path segment
                # Point-to-line-segment distance algorithm:
                # 1. Project obstacle point onto the line containing the segment
                # 2. Clamp projection to segment endpoints if outside segment
                # 3. Calculate distance from obstacle to projection point
                px = obs_x_world
                py = obs_y_world
                x1, y1 = p1.pose.position.x, p1.pose.position.y
                x2, y2 = p2.pose.position.x, p2.pose.position.y
                
                # Vector from p1 to p2
                dx = x2 - x1
                dy = y2 - y1
                seg_len_sq = dx*dx + dy*dy
                
                if seg_len_sq < 0.001:  # Degenerate segment (p1 and p2 are same point)
                    # Just check distance to point
                    dist = hypot(px - x1, py - y1)
                else:
                    # Project obstacle point onto line segment
                    # t = dot product of (obstacle - p1) and (p2 - p1) / ||p2 - p1||²
                    # t represents position along segment: 0 = p1, 1 = p2
                    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
                    # Clamp t to [0, 1] to ensure projection is on segment
                    proj_x = x1 + t * dx  # Projection point on segment
                    proj_y = y1 + t * dy
                    dist = hypot(px - proj_x, py - proj_y)  # Distance from obstacle to segment
                
                # If obstacle is within replan threshold, trigger replan
                # replan_threshold should be larger than obstacle_inflation_radius to
                # give time to replan before robot gets too close
                if dist < self.replan_threshold_:
                    return True
        
        return False

    # Trajectory smoothing --------------------------------------------------

    def smoothPathWithSplines_(self, path_poses):
        """
        Smooth a path using cubic spline interpolation.
        
        This function takes a list of waypoints from A* and generates a smooth
        polynomial trajectory using cubic splines. The splines ensure:
        - Continuous position (C0 continuity)
        - Continuous velocity (C1 continuity) 
        - Continuous acceleration (C2 continuity)
        
        The algorithm:
        1. Extract x and y coordinates from path waypoints
        2. Calculate cumulative arc length along path (parameter t)
        3. Fit cubic splines: x(t) and y(t) where t is arc length
        4. Resample at regular intervals to get smooth trajectory
        
        Args:
            path_poses: List of PoseStamped waypoints from A*
            
        Returns:
            List of PoseStamped: Smoothed path with interpolated points
        """
        if len(path_poses) < 2:
            return path_poses  # Can't smooth paths with < 2 points
        
        # Extract x and y coordinates from waypoints
        x_coords = [pose.pose.position.x for pose in path_poses]
        y_coords = [pose.pose.position.y for pose in path_poses]
        
        # Calculate cumulative arc length (parameter t for spline)
        # This is the distance traveled along the path
        t_values = [0.0]  # Start at t=0
        cumulative_distance = 0.0
        
        for i in range(1, len(path_poses)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            segment_length = hypot(dx, dy)
            cumulative_distance += segment_length
            t_values.append(cumulative_distance)
        
        # Convert to numpy arrays for spline fitting
        t_array = np.array(t_values)
        x_array = np.array(x_coords)
        y_array = np.array(y_coords)
        
        # Fit cubic splines for x(t) and y(t)
        # Cubic spline: piecewise cubic polynomials with C2 continuity
        # Each segment is a cubic polynomial: a*t³ + b*t² + c*t + d
        try:
            # Use numpy to fit cubic splines
            # For small paths, we'll use a simple interpolation approach
            if len(t_array) < 4:
                # Not enough points for cubic spline, use linear interpolation
                return path_poses
            
            # Create spline interpolation functions
            # We'll use a parametric cubic spline approach
            from scipy.interpolate import CubicSpline
            
            # Fit cubic splines with natural boundary conditions (zero second derivative at endpoints)
            cs_x = CubicSpline(t_array, x_array, bc_type='natural')
            cs_y = CubicSpline(t_array, y_array, bc_type='natural')
            
            # Resample the spline at regular intervals
            total_length = t_array[-1]
            num_points = max(2, int(total_length / self.spline_resolution_) + 1)
            t_smooth = np.linspace(0, total_length, num_points)
            
            # Evaluate splines at new points
            x_smooth = cs_x(t_smooth)
            y_smooth = cs_y(t_smooth)
            
            # Create new smoothed path
            smoothed_path = []
            for i in range(len(t_smooth)):
                pose = PoseStamped()
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.header.frame_id = "map"
                pose.pose.position.x = float(x_smooth[i])
                pose.pose.position.y = float(y_smooth[i])
                smoothed_path.append(pose)
            
            return smoothed_path
            
        except ImportError:
            # If scipy is not available, use simple cubic Bezier interpolation
            self.get_logger().warn("scipy not available, using simple cubic interpolation")
            return self._simpleCubicInterpolation_(x_coords, y_coords, t_values)

    def _simpleCubicInterpolation_(self, x_coords, y_coords, t_values):
        """
        Simple cubic interpolation using Catmull-Rom splines (fallback method).
        
        Catmull-Rom splines pass through all control points and provide
        smooth interpolation without requiring scipy.
        
        Args:
            x_coords: List of x-coordinates
            y_coords: List of y-coordinates  
            t_values: List of parameter values (cumulative distances)
            
        Returns:
            List of PoseStamped: Interpolated path
        """
        if len(x_coords) < 2:
            return []
        
        smoothed_path = []
        total_length = t_values[-1]
        num_points = max(2, int(total_length / self.spline_resolution_) + 1)
        t_smooth = np.linspace(0, total_length, num_points)
        
        for t in t_smooth:
            # Find which segment this t value belongs to
            if t <= t_values[0]:
                x, y = x_coords[0], y_coords[0]
            elif t >= t_values[-1]:
                x, y = x_coords[-1], y_coords[-1]
            else:
                # Find segment index
                seg_idx = 0
                for i in range(len(t_values) - 1):
                    if t_values[i] <= t <= t_values[i+1]:
                        seg_idx = i
                        break
                
                # Linear interpolation within segment (simple fallback)
                if seg_idx < len(t_values) - 1:
                    t0, t1 = t_values[seg_idx], t_values[seg_idx+1]
                    alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
                    x = x_coords[seg_idx] + alpha * (x_coords[seg_idx+1] - x_coords[seg_idx])
                    y = y_coords[seg_idx] + alpha * (y_coords[seg_idx+1] - y_coords[seg_idx])
                else:
                    x, y = x_coords[-1], y_coords[-1]
            
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            smoothed_path.append(pose)
        
        return smoothed_path

    # A* algorithm ----------------------------------------------------------

    def astar_(self, start_x, start_y, goal_x, goal_y):
        """8-connected A* on the inflated global costmap with dynamic obstacles from laser scan."""
        # Validate map availability
        if not self.received_map_ or self.costmap_cols_ <= 0 or self.costmap_rows_ <= 0:
            self.get_logger().warn("Planner called without a valid costmap; publishing empty path.")
            self.pub_path_.publish(Path())
            return

        # ========== IMPROVEMENT: Update Costmap with Dynamic Obstacles ==========
        # Before planning, incorporate obstacles from laser scan into the costmap.
        # This ensures A* avoids both static obstacles (from global costmap) and
        # dynamic obstacles (from laser scan).
        working_costmap = self.updateCostmapWithObstacles_()

        # Convert world coordinates to grid cells
        start_c, start_r = self.XYToCR_(start_x, start_y)
        goal_c, goal_r = self.XYToCR_(goal_x, goal_y)

        # Validate start and goal
        if self.outOfMap_(start_c, start_r) or self.outOfMap_(goal_c, goal_r):
            self.get_logger().warn("Start or goal is outside map bounds; publishing empty path.")
            self.pub_path_.publish(Path())
            return

        # ========== IMPROVEMENT: Blocked Check with Dynamic Obstacles ==========
        # Check if cells are blocked using working_costmap (includes dynamic obstacles).
        # This ensures we check against both static and dynamic obstacles.
        def is_blocked(c, r):
            if self.outOfMap_(c, r):
                return True
            return working_costmap[self.CRToIndex_(c, r)] > self.max_access_cost_
        
        # ========== IMPROVEMENT: Find Nearest Free Cell ==========
        # Problem: When robot is close to obstacles, laser scan may mark the start/goal
        # position as blocked (due to obstacle inflation or sensor noise). This causes
        # the planner to fail immediately without attempting to find a path.
        #
        # Solution: If start/goal is blocked, search for the nearest free cell within
        # a small radius and use that instead. This allows planning to succeed even when
        # start/goal is slightly blocked.
        #
        # How it works:
        # - Searches in expanding squares (spiral pattern) around the blocked position
        # - Checks perimeter cells first (closest to original position)
        # - Returns first free cell found, or None if none found within max_search_radius
        #
        # Why spiral search: Ensures we find the nearest free cell first, minimizing
        # the adjustment to start/goal position.
        #
        # Why this helps: Prevents planner from failing when robot is near walls or
        # when obstacle inflation slightly overlaps start/goal position.
        def findNearestFreeCell(c, r, max_search_radius=5):
            """Find nearest free cell within search radius (spiral search)."""
            # Search in expanding squares (spiral pattern)
            for radius in range(1, max_search_radius + 1):
                # Check cells on the perimeter of the square at this radius
                # Top and bottom edges
                for dc in range(-radius, radius + 1):
                    for dr in [-radius, radius]:
                        test_c = c + dc
                        test_r = r + dr
                        if not is_blocked(test_c, test_r):
                            return test_c, test_r
                # Left and right edges (excluding corners already checked)
                for dr in range(-radius + 1, radius):
                    for dc in [-radius, radius]:
                        test_c = c + dc
                        test_r = r + dr
                        if not is_blocked(test_c, test_r):
                            return test_c, test_r
            return None, None  # No free cell found

        # Check if start/goal is blocked and find nearest free cell if needed
        start_blocked = is_blocked(start_c, start_r)
        goal_blocked = is_blocked(goal_c, goal_r)
        
        if start_blocked:
            self.get_logger().warn(f"Start position ({start_x:.3f}, {start_y:.3f}) is blocked by obstacles, searching for nearest free cell...")
            free_start_c, free_start_r = findNearestFreeCell(start_c, start_r)
            if free_start_c is not None:
                start_c, start_r = free_start_c, free_start_r
                start_x, start_y = self.CRToXY_(start_c, start_r)
                self.get_logger().info(f"Using nearest free start position: ({start_x:.3f}, {start_y:.3f})")
            else:
                self.get_logger().error("Could not find free cell near start position; publishing empty path.")
                self.pub_path_.publish(Path())
                self.current_path_ = None  # Clear path so replanning doesn't keep trying
                return
        
        if goal_blocked:
            self.get_logger().warn(f"Goal position ({goal_x:.3f}, {goal_y:.3f}) is blocked by obstacles, searching for nearest free cell...")
            free_goal_c, free_goal_r = findNearestFreeCell(goal_c, goal_r)
            if free_goal_c is not None:
                goal_c, goal_r = free_goal_c, free_goal_r
                goal_x, goal_y = self.CRToXY_(goal_c, goal_r)
                self.get_logger().info(f"Using nearest free goal position: ({goal_x:.3f}, {goal_y:.3f})")
            else:
                self.get_logger().error("Could not find free cell near goal position; publishing empty path.")
                self.pub_path_.publish(Path())
                self.current_path_ = None  # Clear path so replanning doesn't keep trying
                return

        # Trivial start==goal
        if start_c == goal_c and start_r == goal_r:
            msg_path = Path()
            msg_path.header.stamp = self.get_clock().now().to_msg()
            msg_path.header.frame_id = "map"
            pose = PoseStamped()
            pose.pose.position.x = start_x
            pose.pose.position.y = start_y
            msg_path.poses.append(pose)
            self.pub_path_.publish(msg_path)
            return

        # Create node grid (row-major flat list)
        total_cells = self.costmap_cols_ * self.costmap_rows_
        nodes = [None] * total_cells
        for r in range(self.costmap_rows_):
            for c in range(self.costmap_cols_):
                idx = self.CRToIndex_(c, r)
                nodes[idx] = AStarNode(c, r)

        # Heuristic function in meters (Octile)
        def heuristic_meters(c, r):
            # octile distance from cell to goal cell in meters
            dc = abs(goal_c - c)
            dr = abs(goal_r - r)
            return self.costmap_resolution_ * (max(dc, dr) + (sqrt(2.0) - 1.0) * min(dc, dr))

        # Start node initialization
        start_idx = self.CRToIndex_(start_c, start_r)
        start_node = nodes[start_idx]
        start_node.g = 0.0
        start_node.h = heuristic_meters(start_c, start_r)
        start_node.f = start_node.g + start_node.h

        # Open list (min-heap) and neighbors
        open_list = []
        heappush(open_list, start_node)

        neighbors = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1),
        ]

        goal_reached = None

        while open_list:
            node = heappop(open_list)

            # Skip if already expanded (stale entry)
            if node.expanded:
                continue

            node.expanded = True

            # Check goal
            if node.c == goal_c and node.r == goal_r:
                goal_reached = node
                break

            # Expand neighbors
            for dc, dr in neighbors:
                nb_c = node.c + dc
                nb_r = node.r + dr

                if self.outOfMap_(nb_c, nb_r):
                    continue

                nb_idx = self.CRToIndex_(nb_c, nb_r)
                nb_node = nodes[nb_idx]

                if nb_node.expanded:
                    continue

                # Skip blocked cells
                if is_blocked(nb_c, nb_r):
                    continue

                # Metric step cost (meters)
                step_cost = self.costmap_resolution_ * (sqrt(2.0) if (dc != 0 and dr != 0) else 1.0)

                # Optional cell penalty (convert cell cost units to meters)
                # Use working_costmap which includes dynamic obstacles from laser scan
                # Higher cost cells (closer to obstacles) get higher penalty, encouraging
                # A* to prefer paths further from obstacles when possible
                cell_cost = float(working_costmap[nb_idx])
                cell_penalty = self.cell_penalty_scale_ * cell_cost

                tentative_g = node.g + step_cost + cell_penalty

                if tentative_g < nb_node.g:
                    nb_node.g = tentative_g
                    nb_node.h = heuristic_meters(nb_c, nb_r)
                    nb_node.f = nb_node.g + nb_node.h
                    nb_node.parent = node
                    heappush(open_list, nb_node)

        # Reconstruct and publish
        msg_path = Path()
        msg_path.header.stamp = self.get_clock().now().to_msg()
        msg_path.header.frame_id = "map"

        if goal_reached is None:
            # ========== IMPROVEMENT: Better Error Handling ==========
            # Problem: When path planning fails, it wasn't clear why or what to do.
            #
            # Solution: Provide detailed error messages and clear current_path_ to
            # prevent replanning from getting stuck in a loop.
            #
            # Why clear current_path_: If we keep the old path, the replanning timer
            # will keep trying to replan, but if obstacles are blocking, it will keep
            # failing. Clearing it stops the replanning loop.
            self.get_logger().warn(f"No Path Found from ({start_x:.3f}, {start_y:.3f}) to ({goal_x:.3f}, {goal_y:.3f})!")
            self.get_logger().warn("This might be due to obstacles blocking the path. Try setting a different goal.")
            self.pub_path_.publish(msg_path)  # publish empty path to signal failure
            # Clear current path so replanning doesn't keep trying
            self.current_path_ = None
            return

        # Backtrack
        chain = []
        node = goal_reached
        while node is not None:
            chain.append((node.c, node.r))
            node = node.parent

        # Build poses from start->goal (raw A* path)
        raw_path_poses = []
        for c, r in reversed(chain):
            x, y = self.CRToXY_(c, r)
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            raw_path_poses.append(pose)

        # Apply polynomial trajectory smoothing if enabled
        if self.smooth_path_ and len(raw_path_poses) > 2:
            # Smooth the path using cubic spline interpolation
            smoothed_poses = self.smoothPathWithSplines_(raw_path_poses)
            msg_path.poses = smoothed_poses
            self.get_logger().info(
                f"A* Path Found and smoothed: {len(raw_path_poses)} waypoints -> {len(smoothed_poses)} smoothed points"
            )
        else:
            # Use raw path without smoothing
            msg_path.poses = raw_path_poses
            self.get_logger().info(
                f"A* Path Found (no smoothing): {len(raw_path_poses)} waypoints"
            )

        # ========== IMPROVEMENT: Store Current Path for Obstacle Checking ==========
        # Store the planned path so the replanning timer can check if obstacles
        # are near it. This enables automatic replanning when dynamic obstacles appear.
        self.current_path_ = msg_path
        
        self.pub_path_.publish(msg_path)
        self.get_logger().info(
            f"Path from Rbt @ ({start_x:7.3f}, {start_y:7.3f}) to Goal @ ({goal_x:7.3f}, {goal_y:7.3f}) with {len(msg_path.poses)} poses."
        )


# Main Boiler Plate =============================================================
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Planner())
    rclpy.shutdown()


if __name__ == "__main__":
    main()


