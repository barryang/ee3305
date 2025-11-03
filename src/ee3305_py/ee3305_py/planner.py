from heapq import heappush, heappop
from math import hypot, floor, inf, sqrt

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    qos_profile_services_default,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path


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

        # Parameters: Get Values
        self.max_access_cost_ = self.get_parameter("max_access_cost").value
        self.cell_penalty_scale_ = self.get_parameter("cell_penalty_scale").value

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

        # Path publisher
        self.pub_path_ = self.create_publisher(
            Path,
            "path",
            10
        )

        # Timer
        self.timer = self.create_timer(0.1, self.callbackTimer_)

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

    def callbackSubGlobalCostmap_(self, msg: OccupancyGrid):
        """Latches the global costmap (inflated costs) and its metadata."""
        self.costmap_ = list(msg.data)
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

    def callbackTimer_(self):
        if not self.received_map_ or not self.has_new_request_:
            return

        # run the path planner (A*)
        self.astar_(self.rbt_x_, self.rbt_y_, self.goal_x_, self.goal_y_)
        self.has_new_request_ = False

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

    # A* algorithm ----------------------------------------------------------

    def astar_(self, start_x, start_y, goal_x, goal_y):
        """8-connected A* on the inflated global costmap."""
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

        # Blocked check
        def is_blocked(c, r):
            return self.costmap_[self.CRToIndex_(c, r)] > self.max_access_cost_

        if is_blocked(start_c, start_r) or is_blocked(goal_c, goal_r):
            self.get_logger().warn("Start or goal cell exceeds max_access_cost; publishing empty path.")
            self.pub_path_.publish(Path())
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

        # Heuristic function in meters (Euclidean)
        def heuristic_meters(c, r):
            # distance from cell center to goal cell center in meters
            dc = (goal_c - c)
            dr = (goal_r - r)
            return hypot(dc, dr) * self.costmap_resolution_

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
                cell_cost = float(self.costmap_[nb_idx])
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
            self.get_logger().warn("No Path Found!")
            self.pub_path_.publish(msg_path)  # publish empty path to signal failure
            return

        # Backtrack
        chain = []
        node = goal_reached
        while node is not None:
            chain.append((node.c, node.r))
            node = node.parent

        # Build poses from start->goal
        for c, r in reversed(chain):
            x, y = self.CRToXY_(c, r)
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            msg_path.poses.append(pose)

        self.pub_path_.publish(msg_path)
        self.get_logger().info(
            f"A* Path Found from Rbt @ ({start_x:7.3f}, {start_y:7.3f}) to Goal @ ({goal_x:7.3f}, {goal_y:7.3f}) with {len(msg_path.poses)} poses."
        )


# Main Boiler Plate =============================================================
def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(Planner())
    rclpy.shutdown()


if __name__ == "__main__":
    main()

