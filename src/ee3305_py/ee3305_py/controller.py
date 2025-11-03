from math import hypot, atan2, inf, cos, sin

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


class Controller(Node):

    def __init__(self, node_name="controller"):
        # Node Constructor =============================================================
        super().__init__(node_name)

        # Parameters: Declare
        self.declare_parameter("frequency", float(20))
        self.declare_parameter("lookahead_distance", float(0.3))
        self.declare_parameter("lookahead_lin_vel", float(0.1))
        self.declare_parameter("stop_thres", float(0.1))
        self.declare_parameter("max_lin_vel", float(0.2))
        self.declare_parameter("max_ang_vel", float(2.0))


        # parameters: declase user

        self.declare_parameter("curvature_threshold", float(1.8))
        self.declare_parameter("proximity_threshold", float(0.12))
        self.declare_parameter("lookahead_gain", float(1.4))

        # Parameters: Get Values
        self.frequency_ = self.get_parameter("frequency").value
        self.lookahead_distance_ = self.get_parameter("lookahead_distance").value
        self.lookahead_lin_vel_ = self.get_parameter("lookahead_lin_vel").value
        self.stop_thres_ = self.get_parameter("stop_thres").value
        self.max_lin_vel_ = self.get_parameter("max_lin_vel").value
        self.max_ang_vel_ = self.get_parameter("max_ang_vel").value


        #parameters: get values user
        self.curvature_threshold =  self.get_parameter("curvature_threshold").value
        self.proximity_threshold = self.get_parameter("proximity_threshold").value
        self.lookahead_gain = self.get_parameter("lookahead_gain").value



        # Handles: Topic Subscribers
        # !TODO: path subscriber
        self.sub_path_ = self.create_subscription(
            Path,
            "path",
            self.callbackSubPath_,
            10,
        )
        # !TODO: odometry subscriber
        self.sub_odom_ = self.create_subscription(
            Odometry,
            "odom",
            self.callbackSubOdom_,
            10,
        )
        # Handles: Topic Publishers
        # !TODO: command velocities publisher
        self.pub_cmd_vel_ = self.create_publisher(
            TwistStamped, 
            "cmd_vel", 
            10
        )
        # !TODO: lookahead point publisher
        self.pub_lookahead_ = self.create_publisher(
            PoseStamped, 
            "lookahead", 
            10
        )
        # Laser scan subscriber
        self.sub_scan_ = self.create_subscription(
            LaserScan,
            "scan",
            self.callbackSubScan_,
            qos_profile_sensor_data,
        )
        # Handles: Timers
        self.timer = self.create_timer(1.0 / self.frequency_, self.callbackTimer_)

        # Other Instance Variables
        self.received_odom_ = False
        self.received_path_ = False
        self.path_count = 0
        self.lookahead_found = False
        self.received_scan_ = False
        self.current_scan_ = None

    # Callbacks =============================================================
    
    # Path subscriber callback
    def callbackSubPath_(self, msg: Path):
        if len(msg.poses) == 0:  # not msg.poses is fine but not clear
            self.get_logger().warn(f"Received path message is empty!")
            return  # do not update the path if no path is returned. This will ensure the copied path contains at least one point when the first non-empty path is received.

        # !TODO: copy the array from the path
        self.path_poses_ = msg.poses
        self.path_count = 0
        self.received_path_ = True

    # Odometry subscriber callback
    def callbackSubOdom_(self, msg: Odometry):
        # !TODO: write robot pose to rbt_x_, rbt_y_, rbt_yaw_
        self.rbt_x_ = msg.pose.pose.position.x
        self.rbt_y_ = msg.pose.pose.position.y
        
        #the robots pose is in quarternion space (crazy space)
        q_w = msg.pose.pose.orientation.w
        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z

        change_y = 2 * (q_w*q_z + q_x*q_y)
        change_x = 1 - 2*(q_y**2 + q_z**2)
        
        self.rbt_yaw_ = atan2(change_y, change_x)

        self.received_odom_ = True

    # Gets the lookahead point's coordinates based on the current robot's position and planner's path
    # Make sure path and robot positions are already received, and the path contains at least one point.
    def getLookaheadPoint_(self):
        # Find the point along the path that is closest to the robot
        #assumes self.path_poses[0] is the first closes position

        print("getting closest point")
        closest_dist = 100000000000000
        closest_point_x = 100000000000000
        closest_point_y = 100000000000000
        closest_point_index = 0
        # might want to change to binary search, maybe buggy cos the array is sorted in steps in time not position
        # maybe just chatgpt?
        # other ways that use memoisation have assumptions that could maybe buggy
        for i, j in enumerate(self.path_poses_):
            distance = hypot(j.pose.position.x - self.rbt_x_, j.pose.position.y - self.rbt_y_)
            if distance < closest_dist:
                closest_dist = distance
                closest_point_x = j.pose.position.x
                closest_point_y = j.pose.position.y 
                closest_point_index = i
        

        print("got closest point:" +  str(closest_point_x) + ", " + str(closest_point_y) + "\n at index: " + str(closest_point_index))

        self.lookahead_found = False
        for i, j in enumerate(self.path_poses_[closest_point_index:]):
            distance = hypot(j.pose.position.x - closest_point_x, j.pose.position.y - closest_point_y)
            if distance > self.lookahead_distance_:
                self.get_logger().info(f"distance: {distance:.3f}")
                self.path_count = i
                lookahead_x = j.pose.position.x
                lookahead_y = j.pose.position.y
                self.lookahead_found = True
                print("lookahead found")
                break

        if not self.lookahead_found:
            print("lookahead not found")
            # From the closest point, iterate towards the goal and find the first point that is at least a lookahead distance away.
            # Return the goal point if no such lookahead point can be found
            lookahead_idx = len(self.path_poses_) - 1
            # Get the lookahead coordinates
            lookahead_pose = self.path_poses_[lookahead_idx]
            lookahead_x = lookahead_pose.pose.position.x
            lookahead_y = lookahead_pose.pose.position.y
            print("using goal position")

        # Publish the lookahead coordinates
        msg_lookahead = PoseStamped()
        msg_lookahead.header.stamp = self.get_clock().now().to_msg()
        msg_lookahead.header.frame_id = "map"
        msg_lookahead.pose.position.x = lookahead_x
        msg_lookahead.pose.position.y = lookahead_y
        self.pub_lookahead_.publish(msg_lookahead)
        print("looking ahead")
        # Return the coordinates
        return lookahead_x, lookahead_y

     # Laser scan subscriber callback
    def callbackSubScan_(self, msg: LaserScan):
        """Stores the latest laser scan for obstacle detection."""
        self.current_scan_ = msg
        self.received_scan_ = True

    # Find the closest obstacle distance from laser scan
    def getClosestObstacleDistance_(self):
        """
        Returns the distance to the closest obstacle detected by the laser scan.
        Returns None if no valid scan data is available.
        """
        if not self.received_scan_ or self.current_scan_ is None:
            return None
        
        scan = self.current_scan_
        min_distance = inf
        
        # Iterate through all laser scan ranges
        for range_val in scan.ranges:
            # Filter out invalid readings
            if (range_val < scan.range_min or 
                range_val > scan.range_max or 
                range_val == inf or
                range_val != range_val):  # NaN check
                continue
            
            # Update minimum distance if this reading is closer
            if range_val < min_distance:
                min_distance = range_val
        
        # Return None if no valid readings found, otherwise return minimum distance
        return min_distance if min_distance != inf else None
    
    # Implement the pure pursuit controller here
    def callbackTimer_(self):
        if not self.received_odom_ or not self.received_path_:
            return  # return silently if path or odom is not received.

        # get lookahead point
        lookahead_x, lookahead_y = self.getLookaheadPoint_()

        # get distance to lookahead point (not to be confused with lookahead_distance)
        # assume point_x and point_y contains a point's coordinates
        change_x = lookahead_x - self.rbt_x_
        change_y = lookahead_y - self.rbt_y_
        lookahead_point_distance = hypot(change_x, change_y)
        
        #robot frame x and y coordinates
        x_rbt_frame =  change_x*cos(self.rbt_yaw_) + change_y*sin(self.rbt_yaw_)
        y_rbt_frame = change_y*cos(self.rbt_yaw_) - change_x*sin(self.rbt_yaw_)
        
        movement_rbt = hypot(x_rbt_frame, y_rbt_frame)

        # stop the robot if close to the point.
        if movement_rbt < self.stop_thres_:
            lin_vel = 0.0
            ang_vel = 0.0
            print("lookahead distance: " + str(lookahead_point_distance))
            print("robot is too close to the lookahead point")
        else:
        # get curvature
            c = 2*y_rbt_frame/(movement_rbt**2)
            
        # calculate velocities
            lin_vel = self.lookahead_lin_vel_
            ang_vel = c*lin_vel
        # curvature heuristic
            if self.curvature_threshold < c:
                v_c = lin_vel * self.curvature_threshold / c
                print("using curvature heuristic: " + str(v_c))
            else:
                v_c = lin_vel
                print("using normal linear velocity without curve heuristic")
            
        
        # proximity heuristic
            d_0 = self.getClosestObstacleDistance_()
            print("closet_obstacle: " + str(d_0))
            print("proximity_threshold: " + str(self.proximity_threshold))
            if d_0 == None:
                v = v_c
                print("using normal linear velocity without prox")
            elif d_0 < self.proximity_threshold:
                v = v_c * d_0 / self.proximity_threshold
                print("using proximity heuristic: " + str(v))
            else:
                v = v_c
                print("using normal linear velocity without prox")
            
        # vary lookahead
            L_h = v * self.lookahead_gain

            self.lookahead_distance_ = L_h


            lin_vel = v
            ang_vel = c*lin_vel
        # saturate velocities. The following can result in the wrong curvature,
        # but only when the robot is travelling too fast (which should not occur if well tuned).
            if lin_vel > self.max_lin_vel_ :
                lin_vel = self.max_lin_vel_
            if ang_vel > self.max_ang_vel_:
                ang_vel = ang_vel
        print("linear velocity: " + str(lin_vel))
        print("angular velocity: " + str(ang_vel))
        # publish velocities
        msg_cmd_vel = TwistStamped()
        msg_cmd_vel.header.stamp = self.get_clock().now().to_msg()
        msg_cmd_vel.twist.linear.x = lin_vel
        msg_cmd_vel.twist.angular.z = ang_vel
        self.pub_cmd_vel_.publish(msg_cmd_vel)


# Main Boiler Plate =============================================================
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Controller())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
