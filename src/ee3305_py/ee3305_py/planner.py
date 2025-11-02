from heapq import heappush, heappop
from math import hypot, floor, inf, sqrt

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    qos_profile_services_default,
    qos_profile_sensor_data,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class DijkstraNode:
    def __init__(self, c, r):
        self.parent = None
        self.f = inf
        self.g = inf
        self.h = inf
        self.c = c
        self.r = r
        self.expanded = False

    def __lt__(self, other):  # comparator for heapq (min-heap) sorting
        return self.g < other.g


class Planner(Node):

    def __init__(self, node_name="planner"):
        # Node Constructor =============================================================
        super().__init__(node_name)

        # Parameters: Declare
        self.declare_parameter("max_access_cost", int(100))

        # Parameters: Get Values
        self.max_access_cost_ = self.get_parameter("max_access_cost").value

        # Handles: Topic Subscribers
        # Global costmap subscriber
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

        # !TODO: Path request subscriber
        self.sub_path_request = self.create_subscription(
            Path,
            "path_request",
            self.callbackSubPathRequest_,
            10,
        )
        # Laser scan subscriber
        self.sub_scan_ = self.create_subscription(
            LaserScan,
            "scan",
            self.callbackSubScan_,
            qos_profile_sensor_data,
        )
        # Handles: Publishers
        # !TODO: Path publisher
        self.pub_path_ = self.create_publisher(
            Path, 
            "path", 
            10
        )
        # Closest obstacle distance publisher
        self.pub_closest_obstacle_ = self.create_publisher(
            Float32,
            "closest_obstacle_distance",
            10
        )
        # Handles: Timers
        self.timer = self.create_timer(0.1, self.callbackTimer_)

        # Other Instance Variables
        self.has_new_request_ = False
        self.received_map_ = False
        self.received_scan_ = False
        self.current_scan_ = None
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

    # Callbacks =============================================================

    # Path request subscriber callback
    def callbackSubPathRequest_(self, msg: Path):
        """Receives a path request containing two poses: [0]=robot, [1]=goal.
        Copies world coordinates into internal fields and flags a new request.
        """
        if len(msg.poses) < 2:
            self.get_logger().warn("Path request must contain robot and goal poses; ignoring.")
            return

        self.rbt_x_ = msg.poses[0].pose.position.x
        self.rbt_y_ = msg.poses[0].pose.position.y
        self.goal_x_ = msg.poses[1].pose.position.x
        self.goal_y_ = msg.poses[1].pose.position.y
        self.has_new_request_ = True

    # Global costmap subscriber callback
    # This is only run once because the costmap is only published once, at the start of the launch.
    def callbackSubGlobalCostmap_(self, msg: OccupancyGrid):
        """Latches the global costmap (inflated costs) and its metadata.
        The costmap is a flat row-major array of int8 costs in [0..99].
        """
        # Copy data and metadata needed for grid conversions and planning
        self.costmap_ = list(msg.data)
        self.costmap_resolution_ = float(msg.info.resolution)
        self.costmap_origin_x_ = float(msg.info.origin.position.x)
        self.costmap_origin_y_ = float(msg.info.origin.position.y)
        self.costmap_rows_ = int(msg.info.height)
        self.costmap_cols_ = int(msg.info.width)

        # Basic sanity check
        expected_len = self.costmap_rows_ * self.costmap_cols_
        if len(self.costmap_) != expected_len:
            self.get_logger().warn(
                f"Costmap size mismatch (data={len(self.costmap_)}, rows*cols={expected_len}).")

        self.received_map_ = True

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

    # runs the path planner at regular intervals as long as there is a new path request.
    def callbackTimer_(self):
        # Publish closest obstacle distance (runs continuously regardless of path planning)
        closest_obstacle_dist = self.getClosestObstacleDistance_()
        if closest_obstacle_dist is not None:
            msg_distance = Float32()
            msg_distance.data = float(closest_obstacle_dist)
            self.pub_closest_obstacle_.publish(msg_distance)
        else:
            # Publish -1.0 to indicate no valid data
            msg_distance = Float32()
            msg_distance.data = -1.0
            self.pub_closest_obstacle_.publish(msg_distance)

        # Run path planner if there's a new request
        if not self.received_map_ or not self.has_new_request_:
            return  # silently return if no new request or map is not received.

        # run the path planner
        self.dijkstra_(self.rbt_x_, self.rbt_y_, self.goal_x_, self.goal_y_)

        self.has_new_request_ = False

    # Publish the interpolated path for testing
    def publishInterpolatedPath(self, start_x, start_y, goal_x, goal_y):
        msg_path = Path()
        msg_path.header.stamp = self.get_clock().now().to_msg()
        msg_path.header.frame_id = "map"

        dx = start_x - goal_x
        dy = start_y - goal_y
        distance = hypot(dx, dy)
        steps = distance / 0.05

        # Generate poses at every 0.05m
        for i in range(int(steps)):
            pose = PoseStamped()
            pose.pose.position.x = goal_x + dx * i / steps
            pose.pose.position.y = goal_y + dy * i / steps
            msg_path.poses.append(pose)

        # Add the goal pose
        pose = PoseStamped()
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        msg_path.poses.append(pose)

        # Reverse the path (hint)
        msg_path.poses.reverse()

        # publish the path
        self.pub_path_.publish(msg_path)

        self.get_logger().info(
            f"Publishing interpolated path between Start and Goal. Implement dijkstra_() instead."
        )

    # Converts world coordinates to cell column and cell row.
    def XYToCR_(self, x, y):
        """Converts world (x,y) to integer grid column,row (c,r).
        Uses map origin as the bottom-left of cell (0,0) and floors to cell index.
        """
        c = int(floor((x - self.costmap_origin_x_) / self.costmap_resolution_))
        r = int(floor((y - self.costmap_origin_y_) / self.costmap_resolution_))
        return c, r

    # Converts cell column and cell row to world coordinates.
    def CRToXY_(self, c, r):
        """Converts integer grid (c,r) to world coordinates at cell center."""
        x = self.costmap_origin_x_ + (c + 0.5) * self.costmap_resolution_
        y = self.costmap_origin_y_ + (r + 0.5) * self.costmap_resolution_
        return x, y

    # Converts cell column and cell row to flattened array index.
    def CRToIndex_(self, c, r):
        """Converts (c,r) to flat index into row-major costmap array."""
        return r * self.costmap_cols_ + c

    # Returns true if the cell column and cell row is outside the costmap.
    def outOfMap_(self, c, r):
        """Returns True if (c,r) lies outside the map bounds."""
        return (c < 0) or (r < 0) or (c >= self.costmap_cols_) or (r >= self.costmap_rows_)

    # Runs the path planning algorithm based on the world coordinates.
    def dijkstra_(self, start_x, start_y, goal_x, goal_y):
        """8-connected Dijkstra on the inflated global costmap.
        - Blocks cells with cost > max_access_cost_.
        - Uses metric step costs (straight vs diagonal).
        - Reconstructs and publishes a nav_msgs/Path in map frame.
        """
        # self.publishInterpolatedPath(start_x, start_y, goal_x, goal_y)
        # return
        # Validate map availability
        if not self.received_map_ or self.costmap_cols_ <= 0 or self.costmap_rows_ <= 0:
            self.get_logger().warn("Planner called without a valid costmap; publishing empty path.")
            self.pub_path_.publish(Path())
            return

        # Convert world coordinates to grid cells
        start_c, start_r = self.XYToCR_(start_x, start_y)
        goal_c, goal_r = self.XYToCR_(goal_x, goal_y)

        # Validate start and goal
        if self.outOfMap_(start_c, start_r) or self.outOfMap_(goal_c, goal_r):
            self.get_logger().warn("Start or goal is outside map bounds; publishing empty path.")
            self.pub_path_.publish(Path())
            return

        # Reject start/goal if blocked by cost threshold
        def is_blocked(c, r):
            return self.costmap_[self.CRToIndex_(c, r)] > self.max_access_cost_

        if is_blocked(start_c, start_r) or is_blocked(goal_c, goal_r):
            self.get_logger().warn("Start or goal cell exceeds max_access_cost; publishing empty path.")
            self.pub_path_.publish(Path())
            return

        # Trivial case
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
                nodes[idx] = DijkstraNode(c, r)

        # Initialize start node
        start_idx = self.CRToIndex_(start_c, start_r)
        start_node = nodes[start_idx]
        start_node.g = 0.0

        # Min-heap open list by g-cost
        open_list = []
        heappush(open_list, start_node)

        # 8-connected neighbors
        neighbors = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1),
        ]

        # Dijkstra expansion
        goal_reached = None
        while open_list:
            node = heappop(open_list)
            if node.expanded:
                continue
            node.expanded = True

            if node.c == goal_c and node.r == goal_r:
                goal_reached = node
                break

            for dc, dr in neighbors:
                nb_c = node.c + dc
                nb_r = node.r + dr
                if self.outOfMap_(nb_c, nb_r):
                    continue

                nb_idx = self.CRToIndex_(nb_c, nb_r)
                nb_node = nodes[nb_idx]

                if nb_node.expanded:
                    continue

                # Skip blocked cells (above threshold)
                if is_blocked(nb_c, nb_r):
                    continue

                # Step cost in meters (diagonal vs straight)
                step_cost = self.costmap_resolution_ * (sqrt(2.0) if (dc != 0 and dr != 0) else 1.0)

                # Optional mild penalty to bias away from high-cost cells (scaled to meters)
                cell_penalty = 0.0  # could use: 0.001 * self.costmap_[nb_idx]
                tentative_g = node.g + step_cost + cell_penalty

                if tentative_g < nb_node.g:
                    nb_node.g = tentative_g
                    nb_node.parent = node
                    heappush(open_list, nb_node)

        # Reconstruct and publish path if found
        msg_path = Path()
        msg_path.header.stamp = self.get_clock().now().to_msg()
        msg_path.header.frame_id = "map"

        if goal_reached is None:
            self.get_logger().warn("No Path Found!")
            # Publish empty path to signal failure
            self.pub_path_.publish(msg_path)
            return

        # Backtrack from goal to start via parents
        chain = []
        node = goal_reached
        while node is not None:
            chain.append((node.c, node.r))
            node = node.parent

        # Build poses from start->goal (reverse the chain)
        for c, r in reversed(chain):
            x, y = self.CRToXY_(c, r)
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            msg_path.poses.append(pose)

        self.pub_path_.publish(msg_path)
        self.get_logger().info(
            f"Path Found from Rbt @ ({start_x:7.3f}, {start_y:7.3f}) to Goal @ ({goal_x:7.3f}, {goal_y:7.3f}) with {len(msg_path.poses)} poses.")


# Main Boiler Plate =============================================================
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Planner())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
