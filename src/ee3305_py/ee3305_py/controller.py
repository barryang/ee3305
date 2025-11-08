"""
Pure Pursuit Controller Implementation

This controller implements a Pure Pursuit path following algorithm with obstacle avoidance.
The Pure Pursuit algorithm works by:
1. Finding a "lookahead point" on the path that is a certain distance ahead
2. Calculating the curvature needed to reach that point
3. Converting curvature to angular velocity: ω = v * c
4. Applying velocity heuristics for safety (curvature limiting, obstacle avoidance)

Key Formulas:
- Curvature: c = 2*y' / L² where y' is lookahead point's y-coordinate in robot frame, L is distance
- Angular velocity: ω = v * c
- Frame transformation: Rotates world coordinates to robot frame (robot at origin, facing +x)
"""

from math import hypot, atan2, inf, cos, sin, pi

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


class Controller(Node):

    def __init__(self, node_name="controller"):
        """
        Initialize the Pure Pursuit Controller Node
        
        Sets up ROS2 subscribers, publishers, timers, and parameters for the controller.
        The controller subscribes to path, odometry, and laser scan data, then publishes
        velocity commands to move the robot along the path.
        """
        # Node Constructor =============================================================
        super().__init__(node_name)

        # ========== Core Pure Pursuit Parameters ==========
        # Parameters: Declare
        self.declare_parameter("frequency", float(20))  # Controller update rate [Hz]
        self.declare_parameter("lookahead_distance", float(0.2))  # Initial lookahead distance [m]
        self.declare_parameter("lookahead_lin_vel", float(0.1))  # Base linear velocity [m/s]
        self.declare_parameter("stop_thres", float(0.1))  # Stop when this close to goal [m]
        self.declare_parameter("max_lin_vel", float(0.5))  # Maximum allowed linear velocity [m/s]
        self.declare_parameter("max_ang_vel", float(2.0))  # Maximum allowed angular velocity [rad/s]

        # ========== Advanced Navigation Parameters ==========
        # parameters: declare user

        # Curvature threshold: If curvature exceeds this, slow down proportionally
        # Higher values allow sharper turns at higher speeds
        self.declare_parameter("curvature_threshold", float(2.8))  # [1/m] max curvature before slowing
        
        # Proximity threshold: If obstacle is closer than this, reduce velocity linearly
        self.declare_parameter("proximity_threshold", float(0.12))  # [m] obstacle distance threshold
        
        # Lookahead gain: Multiplier for adaptive lookahead (L_h = v * gain)
        # Higher gain = lookahead increases more with speed
        self.declare_parameter("lookahead_gain", float(1.4))  # [] adaptive lookahead multiplier
        
        # Clamping parameters: Prevent lookahead from becoming too small/large (prevents oscillations)
        self.declare_parameter("min_lookahead_distance", float(0.1))  # [m] minimum lookahead
        self.declare_parameter("max_lookahead_distance", float(1.0))  # [m] maximum lookahead
        
        # Velocity smoothing: Exponential filter factor (0-1, higher = smoother but slower response)
        self.declare_parameter("velocity_smoothing_factor", float(0.7))  # [] smoothing factor
        
        # Obstacle detection angle range: Only check obstacles in front/sides (not behind)
        self.declare_parameter("obstacle_angle_range", float(1.57))  # [rad] ±90 degrees default

        # Parameters: Get Values - Store parameter values for use in controller
        self.frequency_ = self.get_parameter("frequency").value
        self.lookahead_distance_ = self.get_parameter("lookahead_distance").value
        self.lookahead_lin_vel_ = self.get_parameter("lookahead_lin_vel").value
        self.stop_thres_ = self.get_parameter("stop_thres").value
        self.max_lin_vel_ = self.get_parameter("max_lin_vel").value
        self.max_ang_vel_ = self.get_parameter("max_ang_vel").value

        # Parameters: Get Values - Advanced navigation parameters
        self.curvature_threshold =  self.get_parameter("curvature_threshold").value
        self.proximity_threshold = self.get_parameter("proximity_threshold").value
        self.lookahead_gain = self.get_parameter("lookahead_gain").value
        self.min_lookahead_distance_ = self.get_parameter("min_lookahead_distance").value
        self.max_lookahead_distance_ = self.get_parameter("max_lookahead_distance").value
        self.velocity_smoothing_factor_ = self.get_parameter("velocity_smoothing_factor").value
        self.obstacle_angle_range_ = self.get_parameter("obstacle_angle_range").value



        # ========== ROS2 Topic Subscribers ==========
        # Handles: Topic Subscribers
        # Subscribe to path from planner (contains waypoints to follow)
        self.sub_path_ = self.create_subscription(
            Path,
            "path",
            self.callbackSubPath_,
            10,
        )
        # Subscribe to odometry (robot's current position and orientation in world frame)
        self.sub_odom_ = self.create_subscription(
            Odometry,
            "odom",
            self.callbackSubOdom_,
            10,
        )
        # Subscribe to laser scan for obstacle detection and avoidance
        self.sub_scan_ = self.create_subscription(
            LaserScan,
            "scan",
            self.callbackSubScan_,
            qos_profile_sensor_data,  # Use sensor QoS for real-time data
        )
        
        # ========== ROS2 Topic Publishers ==========
        # Handles: Topic Publishers
        # Publish velocity commands to move the robot (linear.x and angular.z)
        self.pub_cmd_vel_ = self.create_publisher(
            TwistStamped, 
            "cmd_vel", 
            10
        )
        # Publish lookahead point for visualization in RViz
        self.pub_lookahead_ = self.create_publisher(
            PoseStamped, 
            "lookahead", 
            10
        )
        
        # ========== Timer ==========
        # Handles: Timers
        # Timer that calls the main control loop at specified frequency
        # This is where the Pure Pursuit algorithm runs continuously
        self.timer = self.create_timer(1.0 / self.frequency_, self.callbackTimer_)

        # ========== State Variables ==========
        # Other Instance Variables
        # Flags to track if we've received necessary data
        self.received_odom_ = False  # True when odometry data received
        self.received_path_ = False  # True when path data received
        self.received_scan_ = False  # True when laser scan data received
        
        # Path tracking variables
        self.path_count = 0  # Index tracker for path following optimization
        self.lookahead_found = False  # Flag if lookahead point was found on path
        self.current_scan_ = None  # Store latest laser scan message
        
        # Velocity smoothing variables (for exponential filter)
        # Store previous velocities to smooth out jerky motion
        self.last_lin_vel_ = 0.0  # Previous linear velocity [m/s]
        self.last_ang_vel_ = 0.0  # Previous angular velocity [rad/s]

    # ========================================================================
    # Callbacks - Functions called when ROS2 messages are received
    # ========================================================================
    
    def callbackSubPath_(self, msg: Path):
        """
        Callback function called when a new path is received from the planner.
        
        The path contains a list of waypoints (poses) that the robot should follow.
        This function stores the path and resets path tracking state.
        
        Args:
            msg: Path message containing list of PoseStamped waypoints
        """
        # Validate path is not empty
        if len(msg.poses) == 0:
            self.get_logger().warn(f"Received path message is empty!")
            return  # Don't update path if empty - keep previous path

        # Store the path waypoints (list of PoseStamped messages)
        self.path_poses_ = msg.poses
        self.path_count = 0  # Reset path index tracker
        self.received_path_ = True  # Mark that we have a valid path

    def callbackSubOdom_(self, msg: Odometry):
        """
        Callback function called when new odometry data is received.
        
        Extracts robot's position (x, y) and orientation (yaw angle) from the odometry message.
        The orientation is stored as a quaternion in ROS2, so we convert it to yaw angle.
        
        Args:
            msg: Odometry message containing robot's pose in world frame
        """
        # Extract position (in world/map frame)
        self.rbt_x_ = msg.pose.pose.position.x  # Robot x-coordinate [m]
        self.rbt_y_ = msg.pose.pose.position.y  # Robot y-coordinate [m]
        
        # Extract orientation from quaternion and convert to yaw angle
        # ROS2 uses quaternions (w, x, y, z) to represent 3D orientation
        # We need to extract the yaw (rotation around z-axis) for 2D navigation
        q_w = msg.pose.pose.orientation.w
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z

        # Convert quaternion to yaw angle using standard formula
        # This extracts the rotation around the z-axis (vertical axis)
        change_y = 2 * (q_w*q_z + q_x*q_y)  # sin(2*yaw) component
        change_x = 1 - 2*(q_y**2 + q_z**2)  # cos(2*yaw) component
        
        # Calculate yaw angle using atan2 (returns angle in range [-π, π])
        self.rbt_yaw_ = atan2(change_y, change_x)  # Robot heading [rad]

        self.received_odom_ = True  # Mark that we have valid odometry data

    def getLookaheadPoint_(self):
        """
        Find the lookahead point on the path using Pure Pursuit algorithm.
        
        The Pure Pursuit algorithm works by:
        1. Finding the closest point on the path to the robot
        2. Starting from that point, searching forward along the path
        3. Finding the first point that is at least 'lookahead_distance' away
        4. This point becomes the "lookahead point" that the robot will steer towards
        
        Improved: Only considers points ahead of the robot to prevent backward movement.
        
        Returns:
            tuple: (lookahead_x, lookahead_y) - coordinates of lookahead point in world frame
        """
        # ========== Step 1: Find Closest Point on Path ==========
        # Initialize variables to track the closest path point
        closest_dist = inf  # Start with infinity (will find minimum)
        closest_point_x = 0.0
        closest_point_y = 0.0
        closest_point_index = 0
        
        # Iterate through all waypoints in the path
        for i, j in enumerate(self.path_poses_):
            # Calculate distance from robot to this waypoint
            dx = j.pose.position.x - self.rbt_x_  # x-component of vector from robot to waypoint
            dy = j.pose.position.y - self.rbt_y_  # y-component of vector from robot to waypoint
            distance = hypot(dx, dy)  # Euclidean distance (sqrt(dx² + dy²))
            
            # Transform to robot frame to check if point is ahead of robot
            # Robot frame: robot at origin, facing +x direction
            # x_rbt > 0 means point is ahead, x_rbt < 0 means point is behind
            x_rbt = dx*cos(self.rbt_yaw_) + dy*sin(self.rbt_yaw_)
            
            # Only consider points that are ahead (or slightly behind for tolerance)
            # This prevents the robot from selecting points behind it
            if distance < closest_dist and x_rbt > -0.1:  # -0.1m tolerance for slight look-behind
                closest_dist = distance
                closest_point_x = j.pose.position.x
                closest_point_y = j.pose.position.y 
                closest_point_index = i

        # ========== Step 2: Find Lookahead Point ==========
        # Starting from the closest point, search forward along the path
        # to find a point that is at least 'lookahead_distance' away
        self.lookahead_found = False
        for i, j in enumerate(self.path_poses_[closest_point_index:]):
            # Calculate distance from closest point to this waypoint (along path)
            distance = hypot(j.pose.position.x - closest_point_x, 
                           j.pose.position.y - closest_point_y)
            
            # If this point is at least lookahead_distance away, use it as lookahead point
            if distance >= self.lookahead_distance_:
                self.path_count = i  # Track position for optimization
                lookahead_x = j.pose.position.x
                lookahead_y = j.pose.position.y
                self.lookahead_found = True
                break  # Found lookahead point, stop searching

        # ========== Step 3: Fallback to Goal ==========
        # If no point is far enough (e.g., near goal), use the goal point
        if not self.lookahead_found:
            lookahead_idx = len(self.path_poses_) - 1  # Last waypoint is goal
            lookahead_pose = self.path_poses_[lookahead_idx]
            lookahead_x = lookahead_pose.pose.position.x
            lookahead_y = lookahead_pose.pose.position.y

        # Publish lookahead point for visualization in RViz
        msg_lookahead = PoseStamped()
        msg_lookahead.header.stamp = self.get_clock().now().to_msg()
        msg_lookahead.header.frame_id = "map"
        msg_lookahead.pose.position.x = lookahead_x
        msg_lookahead.pose.position.y = lookahead_y
        self.pub_lookahead_.publish(msg_lookahead)
        
        return lookahead_x, lookahead_y

    def callbackSubScan_(self, msg: LaserScan):
        """
        Callback function called when new laser scan data is received.
        
        Stores the latest scan for obstacle detection. The laser scan contains
        an array of distance measurements at different angles around the robot.
        
        Args:
            msg: LaserScan message containing array of range measurements
        """
        self.current_scan_ = msg
        self.received_scan_ = True

    def getClosestObstacleDistance_(self):
        """
        Find the closest obstacle distance from laser scan data.
        
        This function:
        1. Filters out invalid readings (inf, NaN, out of range)
        2. Only considers obstacles in front/sides (not behind robot)
        3. Returns the minimum valid distance found
        
        Returns:
            float: Distance to closest obstacle [m], or None if no valid data
        """
        # Check if scan data is available
        if not self.received_scan_ or self.current_scan_ is None:
            return None
        
        scan = self.current_scan_
        min_distance = inf  # Initialize with infinity (will find minimum)
        angle_min = scan.angle_min  # Starting angle of scan [rad]
        angle_increment = scan.angle_increment  # Angle between each measurement [rad]
        
        # Iterate through all laser scan measurements
        for i, range_val in enumerate(scan.ranges):
            # ========== Filter Invalid Readings ==========
            # Skip readings that are invalid:
            # - Out of sensor range (too close or too far)
            # - Infinity (no obstacle detected)
            # - NaN (invalid measurement)
            if (range_val < scan.range_min or 
                range_val > scan.range_max or 
                range_val == inf or
                range_val != range_val):  # NaN check (NaN != NaN is True)
                continue
            
            # ========== Directional Filtering ==========
            # Only consider obstacles in front/sides of robot (not behind)
            # Calculate the angle of this laser ray relative to robot's front
            angle = angle_min + i * angle_increment
            
            # Skip obstacles outside the specified angle range (behind robot)
            # obstacle_angle_range_ is typically ±90° (1.57 rad)
            if abs(angle) > self.obstacle_angle_range_:
                continue  # Skip obstacles behind robot
            
            # ========== Find Minimum Distance ==========
            # Update minimum distance if this reading is closer
            if range_val < min_distance:
                min_distance = range_val
        
        # Return None if no valid readings found, otherwise return minimum distance
        return min_distance if min_distance != inf else None
    
    def callbackTimer_(self):
        """
        Main Pure Pursuit Control Loop
        
        This function is called at regular intervals (frequency_ Hz) and implements
        the Pure Pursuit path following algorithm. The algorithm:
        
        1. Finds lookahead point on path
        2. Transforms lookahead point to robot frame
        3. Calculates required curvature to reach lookahead point
        4. Applies velocity heuristics (curvature limiting, obstacle avoidance)
        5. Smooths velocities and publishes commands
        
        This is the core control loop that runs continuously while the robot is navigating.
        """
        # ========== Step 0: Check Prerequisites ==========
        # Don't proceed if we don't have necessary data
        if not self.received_odom_ or not self.received_path_:
            return  # Silently return - data not ready yet

        # ========== Step 1: Find Lookahead Point ==========
        # Get the lookahead point on the path (uses current lookahead_distance_)
        # This point is at least lookahead_distance_ away from the closest path point
        lookahead_x, lookahead_y = self.getLookaheadPoint_()

        # ========== Step 2: Transform to Robot Frame ==========
        # Calculate vector from robot to lookahead point (in world frame)
        change_x = lookahead_x - self.rbt_x_  # x-component in world frame [m]
        change_y = lookahead_y - self.rbt_y_  # y-component in world frame [m]
        lookahead_point_distance = hypot(change_x, change_y)  # Distance in world frame [m]
        
        # Transform lookahead point from world frame to robot frame
        # Robot frame: robot at origin (0,0), robot facing +x direction
        # We need to rotate the world-frame vector by -yaw to get robot-frame coordinates
        #
        # Rotation matrix from world to robot frame (rotate by -yaw):
        # [x']   [cos(θ)  sin(θ)] [dx]
        # [y'] = [-sin(θ) cos(θ)] [dy]
        #
        # Where θ is the robot's yaw angle
        x_rbt_frame = change_x*cos(self.rbt_yaw_) + change_y*sin(self.rbt_yaw_)  # Forward component [m]
        y_rbt_frame = change_y*cos(self.rbt_yaw_) - change_x*sin(self.rbt_yaw_)  # Lateral component [m]
        
        # Distance to lookahead point (L in pure pursuit formula)
        # This is the same in any frame, but we use robot frame for calculations
        movement_rbt = hypot(x_rbt_frame, y_rbt_frame)  # L = distance to lookahead [m]
        
        # ========== IMPROVEMENT: Alignment Check Before Moving ==========
        # Problem: Robot moves forward while turning, causing path deviation and collisions.
        #
        # Solution: Require robot to be 90% aligned (facing) the lookahead point before moving forward.
        #
        # How it works:
        # - Calculate angle between robot heading and direction to lookahead point
        # - Check alignment: cos(angle_error) > 0.9 means angle_error < ~26° (90% aligned)
        # - If not aligned, turn in place first; only move forward when aligned
        #
        # Why 90% alignment: cos(26°) ≈ 0.9, meaning robot is facing within 26° of lookahead point
        # This prevents forward motion while turning, reducing path deviation
        angle_to_lookahead = atan2(change_y, change_x)  # Angle to lookahead in world frame
        angle_error = angle_to_lookahead - self.rbt_yaw_  # Angle difference
        
        # Normalize angle error to [-π, π] range
        while angle_error > 3.14159:
            angle_error -= 2 * 3.14159
        while angle_error < -3.14159:
            angle_error += 2 * 3.14159
        
        # 90% alignment check: cos(angle_error) > 0.9 means angle_error < ~0.45 rad (26°)
        alignment_threshold = 0.45  # ~26 degrees for 90% alignment (cos(26°) ≈ 0.9)
        is_aligned = abs(angle_error) < alignment_threshold
        
        # Also check if lookahead is behind robot (x_rbt_frame < 0)
        goal_behind = x_rbt_frame < 0.0

        # ========== Step 3: Check if Robot Should Stop ==========
        # Stop the robot only if:
        # 1. It's very close to the lookahead point AND
        # 2. It's also very close to the goal (last waypoint)
        # This prevents stopping at intermediate waypoints during curves
        
        # Check if we're near the goal (last waypoint)
        goal_pose = self.path_poses_[-1] if len(self.path_poses_) > 0 else None
        near_goal = False
        if goal_pose is not None:
            goal_dist = hypot(goal_pose.pose.position.x - self.rbt_x_,
                             goal_pose.pose.position.y - self.rbt_y_)
            near_goal = goal_dist < self.stop_thres_ * 2.0  # More lenient check for goal proximity
        
        # ========== IMPROVEMENT: Emergency Stop for Obstacles ==========
        # Problem: Robot was crashing into walls because it didn't stop early enough.
        #
        # Solution: Two-tier obstacle avoidance system:
        # 1. Emergency stop (this section): Hard stop if obstacle < 0.15m
        # 2. Gradual slowdown (Step 7): Velocity reduction if obstacle < proximity_threshold
        #
        # How it works:
        # - Get closest obstacle distance from laser scan (d_0)
        # - If obstacle is closer than emergency_stop_threshold (0.15m), trigger emergency stop
        # - Emergency stop: lin_vel = 0, but allow turning away if possible
        #
        # Why 0.15m: This is a critical safety distance. At 0.15m, the robot is very close
        # to collision. Stopping immediately prevents crashes while still allowing escape
        # maneuvers (turning away) if there's lateral clearance.
        #
        # Turning away: If there's lateral offset (y_rbt_frame), robot can turn to
        # avoid the obstacle even while stopped, preparing for a safer path.
        d_0 = self.getClosestObstacleDistance_()
        emergency_stop_threshold = 0.15  # Stop if obstacle closer than this [m]
        emergency_stop = False
        if d_0 is not None and d_0 < emergency_stop_threshold:
            emergency_stop = True
            self.get_logger().warn(f"EMERGENCY STOP: Obstacle at {d_0:.3f}m!")
        
        # Only stop if both conditions are met: close to lookahead AND near goal
        if movement_rbt < self.stop_thres_ and near_goal:
            lin_vel = 0.0  # Stop forward motion
            ang_vel = 0.0  # Stop rotation
            print("lookahead distance: " + str(lookahead_point_distance))
            print("robot reached goal")
        elif emergency_stop:
            # ========== IMPROVEMENT: Emergency Stop Behavior ==========
            # When emergency stop is triggered, we need to:
            # 1. Stop forward motion immediately (safety)
            # 2. Allow turning away if there's escape route (lateral clearance)
            #
            # How it works:
            # - lin_vel = 0: Stop all forward motion to prevent collision
            # - Check lateral offset (y_rbt_frame): If > 0.1m, there's space to turn
            # - Turn direction: If y_rbt_frame > 0, turn left (positive angular velocity)
            #                   If y_rbt_frame < 0, turn right (negative angular velocity)
            # - Turn speed: 50% of max_ang_vel (conservative, allows fine control)
            #
            # Why this works: By allowing controlled turning even when stopped, the robot
            # can reorient itself to find a safer path, rather than being completely stuck.
            lin_vel = 0.0  # Stop forward motion immediately
            # Allow turning away from obstacle if there's lateral clearance
            if abs(y_rbt_frame) > 0.1:  # If there's lateral offset > 0.1m, we can turn
                # Turn in the direction that moves away from obstacle
                # Positive y_rbt_frame = obstacle on right, turn left (positive ang_vel)
                # Negative y_rbt_frame = obstacle on left, turn right (negative ang_vel)
                ang_vel = self.max_ang_vel_ * 0.5 if y_rbt_frame > 0 else -self.max_ang_vel_ * 0.5
            else:
                # No lateral clearance, fully stop (no turning)
                ang_vel = 0.0
            print("EMERGENCY STOP: obstacle too close")
        elif not is_aligned:
            # ========== IMPROVEMENT: Turn to Face Lookahead Point Before Moving ==========
            # Problem: Robot moves forward while turning, causing path deviation and collisions.
            #
            # Solution: Turn in place until robot is 90% aligned (facing) the lookahead point,
            # then allow forward motion. This prevents forward movement while turning.
            #
            # How it works:
            # 1. angle_error is already calculated above (difference between lookahead direction and robot heading)
            # 2. If not aligned (angle_error > alignment_threshold), turn in place
            # 3. Turn in place: lin_vel = 0, ang_vel proportional to angle error
            # 4. Proportional control: Reduce angular velocity when close to target angle
            #
            # Why 90% alignment: cos(26°) ≈ 0.9, meaning robot faces within 26° of lookahead point.
            # This ensures robot is mostly facing the correct direction before moving forward.
            #
            # Why proportional control: When angle_error < 0.5 rad (~29°), we reduce
            # angular velocity proportionally. This prevents overshooting and provides
            # smooth convergence to the target angle.
            
            # Turn in place: no forward motion, only rotation
            lin_vel = 0.0  # No forward motion until aligned
            
            # Set angular velocity based on angle error direction
            # Use 80% of max_ang_vel for fast turning (faster than normal Pure Pursuit)
            if angle_error > 0:
                ang_vel = self.max_ang_vel_ * 0.8  # Turn left (counterclockwise)
            else:
                ang_vel = -self.max_ang_vel_ * 0.8  # Turn right (clockwise)
            
            # Proportional control: reduce angular velocity when close to target angle
            # This prevents overshooting and provides smooth convergence
            if abs(angle_error) < 0.5:  # Within ~29 degrees (0.5 rad)
                # Scale angular velocity proportionally to remaining angle
                # When angle_error → 0, ang_vel → 0 (smooth stop)
                ang_vel *= abs(angle_error) / 0.5  # Proportional reduction
            print(f"Not aligned with lookahead: turning in place, angle_error={angle_error:.3f} rad ({abs(angle_error)*180/3.14159:.1f}°)")
        elif movement_rbt < self.stop_thres_ * 0.5 and not near_goal:
            # If very close to lookahead but not at goal, just reduce velocity slightly
            # Don't stop completely - this allows continuous motion through curves
            print("close to lookahead, reducing velocity slightly")
        else:
            # ========== Step 4: Calculate Curvature ==========
            # Pure Pursuit curvature formula: c = 2*y' / L²
            # where:
            #   c = curvature [1/m] (inverse of turning radius)
            #   y' = lateral offset of lookahead point in robot frame [m]
            #   L = distance to lookahead point [m]
            #
            # The curvature tells us how sharply we need to turn:
            # - Positive curvature = turn left (y' > 0)
            # - Negative curvature = turn right (y' < 0)
            # - Larger |c| = sharper turn
            c = 2*y_rbt_frame/(movement_rbt**2)
            
            # ========== Step 5: Initial Velocity Calculation ==========
            # Start with base linear velocity
            # Angular velocity is calculated from curvature: ω = v * c
            lin_vel = self.lookahead_lin_vel_  # Base linear velocity [m/s]
            ang_vel = c*lin_vel  # Angular velocity from curvature [rad/s]
            
            # ========== Step 6: Curvature-Based Velocity Limiting ==========
            # If the path requires a sharp turn (high curvature), slow down
            # This prevents the robot from trying to turn too sharply at high speed
            # which could cause instability or overshooting
            if abs(c) > self.curvature_threshold:
                # Reduce velocity proportionally to curvature
                # v_c = v_base * (c_threshold / c_actual)
                # This means: sharper turn → lower velocity
                v_c = lin_vel * self.curvature_threshold / abs(c)
                print("using curvature heuristic: " + str(v_c))
            else:
                # Curvature is acceptable, use base velocity
                v_c = lin_vel
                print("using normal linear velocity without curve heuristic")
            
            # ========== Step 7: Obstacle-Based Velocity Limiting ==========
            # If an obstacle is detected nearby, reduce velocity further
            # This provides reactive obstacle avoidance
            # Note: d_0 was already calculated above for emergency stop check
            print("closest_obstacle: " + str(d_0))
            print("proximity_threshold: " + str(self.proximity_threshold))
            
            if d_0 is None:
                # No obstacle data available, use curvature-limited velocity
                v = v_c
                print("using normal linear velocity without prox")
            elif d_0 < self.proximity_threshold:
                # ========== IMPROVEMENT: Improved Obstacle Avoidance ==========
                # Problem: Robot was getting too close to obstacles before slowing down,
                # causing collisions with walls.
                #
                # Solution: More aggressive velocity reduction based on obstacle distance.
                #
                # How it works:
                # - Linear velocity reduction: v = v_c * (d_0 / proximity_threshold)
                # - safety_factor = d_0 / proximity_threshold (ranges from 0 to 1)
                # - When d_0 = proximity_threshold: safety_factor = 1, v = v_c (full speed)
                # - When d_0 → 0: safety_factor → 0, v → 0 (stop)
                #
                # Why this is better:
                # - proximity_threshold was reduced from 0.5m to 0.4m (more conservative)
                # - Robot starts slowing earlier, giving more time to react
                # - max(0.0, ...) ensures safety_factor never goes negative
                # - Combined with emergency stop at 0.15m, provides two-tier safety system
                #
                # Example: If proximity_threshold = 0.4m and obstacle at 0.2m:
                #   safety_factor = 0.2 / 0.4 = 0.5
                #   v = v_c * 0.5 (50% of curvature-limited velocity)
                # Obstacle is too close! Reduce velocity linearly
                # v = v_c * (d_0 / threshold)
                # Closer obstacle → slower speed (safety)
                # When d_0 = threshold, v = v_c (full speed allowed)
                # When d_0 → 0, v → 0 (stop)
                safety_factor = max(0.0, d_0 / self.proximity_threshold)
                v = v_c * safety_factor
                print("using proximity heuristic: " + str(v))
            else:
                # Obstacle is far enough, use curvature-limited velocity
                v = v_c
                print("using normal linear velocity without prox")
            
            # ========== Step 8: Adaptive Lookahead Distance ==========
            # Adjust lookahead distance based on BASE velocity (not reduced velocity)
            # This prevents the lookahead from shrinking when velocity is reduced for curves,
            # which would cause the robot to stop at every curve
            # Use lookahead_lin_vel_ (base velocity) instead of reduced velocity v
            # Higher velocity → larger lookahead (look further ahead)
            # This helps the robot anticipate turns better at higher speeds
            L_h = self.lookahead_lin_vel_ * self.lookahead_gain  # Adaptive lookahead based on base velocity [m]
            
            # Clamp lookahead distance to prevent instability
            # Too small: oscillations, too large: cutting corners
            self.lookahead_distance_ = max(self.min_lookahead_distance_, 
                                          min(L_h, self.max_lookahead_distance_))

            # ========== Step 9: Update Velocities ==========
            # Apply the final velocity (after all heuristics)
            lin_vel = v  # Final linear velocity [m/s]
            ang_vel = c*lin_vel  # Recalculate angular velocity with final linear velocity [rad/s]
            
            # ========== Step 10: Velocity Saturation ==========
            # Enforce maximum velocity limits (safety constraints)
            # Note: Saturating velocities can change the effective curvature,
            # but this is acceptable if parameters are well-tuned
            if lin_vel < 0.0:
                lin_vel = 0.0  # Ensure linear velocity is non-negative (robot can't go backward)
            if lin_vel > self.max_lin_vel_:
                lin_vel = self.max_lin_vel_  # Cap at maximum linear velocity
            
            # ========== IMPROVEMENT: Dynamic Angular Velocity Boost ==========
            # Problem: Robot turning was too slow, especially for sharp turns and
            # when goal was behind. This made navigation inefficient and unresponsive.
            #
            # Solution: Allow higher angular velocity limits for sharp turns.
            #
            # How it works:
            # - Normal case: max_ang_vel_effective = max_ang_vel_ (default limit)
            # - Sharp turn case: If curvature > 1.0 1/m (sharp turn), boost by 20%
            # - Formula: max_ang_vel_effective = max_ang_vel_ * 1.2
            #
            # Why this helps:
            # 1. Sharp turns need faster rotation to complete quickly
            # 2. When goal is behind, robot needs to turn fast to face it
            # 3. 20% boost (1.2x) provides noticeable improvement without being unsafe
            # 4. Only applies to sharp turns (abs(c) > 1.0), normal turns use default limit
            #
            # Why curvature threshold 1.0: Curvature of 1.0 1/m corresponds to a turning
            # radius of 1.0m. Turns sharper than this (smaller radius) benefit from faster
            # angular velocity. This threshold distinguishes gentle curves from sharp turns.
            #
            # Example: If max_ang_vel_ = 3.0 rad/s and curvature = 1.5 1/m:
            #   max_ang_vel_effective = 3.0 * 1.2 = 3.6 rad/s
            #   Robot can turn faster, completing the turn more quickly
            max_ang_vel_effective = self.max_ang_vel_  # Default angular velocity limit
            if abs(c) > 1.0:  # Sharp turn detected (curvature > 1.0 1/m)
                # Boost angular velocity limit by 20% for sharp turns
                # This allows faster, more responsive turning when needed
                max_ang_vel_effective = self.max_ang_vel_ * 1.2
            
            if abs(ang_vel) > max_ang_vel_effective:
                # Cap angular velocity, preserving sign (left/right turn direction)
                ang_vel = max_ang_vel_effective if ang_vel > 0 else -max_ang_vel_effective
        
        # ========== Step 11: Velocity Smoothing ==========
        # Apply exponential smoothing filter to reduce jerky motion
        # This prevents sudden velocity changes that could cause mechanical stress
        # Formula: v_smooth = α * v_last + (1-α) * v_new
        # where α = smoothing_factor (higher = more smoothing, slower response)
        # Only smooth if velocities are changing (not when explicitly stopping)
        if lin_vel != 0.0 or ang_vel != 0.0:
            # Exponential moving average filter
            lin_vel = (self.velocity_smoothing_factor_ * self.last_lin_vel_ + 
                      (1.0 - self.velocity_smoothing_factor_) * lin_vel)
            ang_vel = (self.velocity_smoothing_factor_ * self.last_ang_vel_ + 
                      (1.0 - self.velocity_smoothing_factor_) * ang_vel)
        
        # Store velocities for next iteration's smoothing
        self.last_lin_vel_ = lin_vel
        self.last_ang_vel_ = ang_vel
        
        print("linear velocity: " + str(lin_vel))
        print("angular velocity: " + str(ang_vel))
        
        # ========== Step 12: Publish Velocity Commands ==========
        # Create and publish velocity command message to move the robot
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.header.stamp = self.get_clock().now().to_msg()  # Timestamp
        msg_cmd_vel.twist.linear.x = lin_vel   # Forward velocity [m/s] (positive = forward)
        msg_cmd_vel.twist.angular.z = ang_vel  # Rotational velocity [rad/s] (positive = counterclockwise)
        self.pub_cmd_vel_.publish(msg_cmd_vel)  # Send command to robot


# Main Boiler Plate =============================================================
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Controller())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
