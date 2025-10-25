from threading import Thread
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2
from pymoveit2 import GripperInterface
from pymoveit2.robots import franka as robot
import time


# Helper function to open the gripper
def open_gripper(node, gripper_interface):
    node.get_logger().info(f'Performing gripper action open')
    gripper_interface.open()
    gripper_interface.wait_until_executed()
    time.sleep(2.0)


# Helper function to close the gripper
def close_gripper(node, gripper_interface):
    node.get_logger().info(f'Performing gripper action close')
    gripper_interface.close()
    gripper_interface.wait_until_executed()
    # Sometimes the gripper does not close fully, so we wait a bit
    time.sleep(2.0)

# Helper function to initialise the gripper
def init_gripper(node, gripper_interface):
    node.get_logger().info(f'Initialise gripper')
    open_gripper(node, gripper_interface)
    close_gripper(node, gripper_interface)


# Move the robot to a pose
def move_to_pose(node, moveit2, position, quat_xyzw, cartesian, cartesian_max_step, cartesian_fraction_threshold):
    node.get_logger().info(f"Moving to {{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}")
    moveit2.move_to_pose(position=position, quat_xyzw=quat_xyzw,
                          cartesian=cartesian, cartesian_max_step=cartesian_max_step, 
                          cartesian_fraction_threshold=cartesian_fraction_threshold)
    moveit2.wait_until_executed()
    node.get_logger().info("Moved to position!")


# Box detection callback
def box_detection_callback(node, msg, box_name, box_poses):
    # Update the detected box pose
    box_poses[box_name]['position'] = msg.pose.position
    box_poses[box_name]['orientation'] = msg.pose.orientation
    # node.get_logger().info(f"Updated {box_name} pose: {box_poses[box_name]['position']}")


def pnp(node, moveit2, gripper_interface, box_pose, quat_xyzw, stacking_location):
    # Extract parameters for Cartesian motion
    cartesian = node.get_parameter("cartesian").get_parameter_value().bool_value
    cartesian_max_step = node.get_parameter("cartesian_max_step").get_parameter_value().double_value
    cartesian_fraction_threshold = node.get_parameter("cartesian_fraction_threshold").get_parameter_value().double_value

    def log_movement(action, pose):
        print(f"{action}: x={pose[0]:.3f}, y={pose[1]:.3f}, z={pose[2]:.3f}")

    # --- APPROACH PICK POSITION ---
    approach_pick_pose = [box_pose['position'].x, box_pose['position'].y, box_pose['position'].z + 0.1]
    log_movement("Approaching pick position", approach_pick_pose)
    move_to_pose(node, moveit2, approach_pick_pose, quat_xyzw, cartesian, cartesian_max_step, cartesian_fraction_threshold)

     # --- OPEN GRIPPER (Pick Box) ---
    open_gripper(node, gripper_interface)
    print("Gripper opened - box to be picked")

    # --- MOVE DOWN TO PICK ---
    pick_pose = [box_pose['position'].x, box_pose['position'].y, box_pose['position'].z]
    log_movement("Moving to pick position", pick_pose)
    move_to_pose(node, moveit2, pick_pose, quat_xyzw, cartesian, cartesian_max_step, cartesian_fraction_threshold)

    # --- CLOSE GRIPPER (Pick Up) ---
    close_gripper(node, gripper_interface)
    print("Gripper closed - Box picked up")

    # --- LIFT BOX AFTER PICKING ---
    lift_pose = [box_pose['position'].x, box_pose['position'].y, box_pose['position'].z + 0.15]
    log_movement("Lifting box", lift_pose)
    move_to_pose(node, moveit2, lift_pose, quat_xyzw, cartesian, cartesian_max_step, cartesian_fraction_threshold)

    # --- MOVE TO STACKING LOCATION ---
    approach_place_pose = [stacking_location[0], stacking_location[1], stacking_location[2] + 0.15]
    log_movement("Moving to stacking location", approach_place_pose)
    move_to_pose(node, moveit2, approach_place_pose, quat_xyzw, cartesian, cartesian_max_step, cartesian_fraction_threshold)

    # --- LOWER TO PLACE POSITION ---
    place_pose = [stacking_location[0], stacking_location[1], stacking_location[2]]
    log_movement("Lowering to place position", place_pose)
    move_to_pose(node, moveit2, place_pose, quat_xyzw, cartesian, cartesian_max_step, cartesian_fraction_threshold)

    # --- OPEN GRIPPER (Release Box) ---
    open_gripper(node, gripper_interface)
    print("Gripper opened - Box released")

    # --- MOVE BACK UP AFTER PLACING ---
    retreat_pose = [stacking_location[0], stacking_location[1], stacking_location[2] + 0.15]
    log_movement("Moving up after placing", retreat_pose)
    move_to_pose(node, moveit2, retreat_pose, quat_xyzw, cartesian, cartesian_max_step, cartesian_fraction_threshold)

    # --- UPDATE STACKING HEIGHT FOR NEXT BOX ---
    stacking_location[2] += 0.05
    print(f"Updated stacking height to: {stacking_location[2]:.3f}\n")


def main():
    rclpy.init()

    node = Node("pnp")

    node.declare_parameter("planner_id", "RRTConnectkConfigDefault")
    node.declare_parameter("cartesian", False)
    node.declare_parameter("cartesian_max_step", 0.0025)
    node.declare_parameter("cartesian_fraction_threshold", 0.0)
    node.declare_parameter("cartesian_jump_threshold", 0.0)
    node.declare_parameter("cartesian_avoid_collisions", False)

    callback_moveit_group = ReentrantCallbackGroup()
    callback_gripper_group = ReentrantCallbackGroup()

    # Create MoveIt 2 interface
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=robot.base_link_name(),
        end_effector_name=robot.end_effector_name(),
        group_name=robot.MOVE_GROUP_ARM,
        callback_group=callback_moveit_group,
    )
    # Add a delay to ensure MoveIt 2 is ready
    node.get_logger().info("Waiting for MoveIt 2 to be ready...")
    time.sleep(2)  # Allow time for MoveIt 2 initialization
    moveit2.planner_id = (
        node.get_parameter("planner_id").get_parameter_value().string_value
    )


    gripper_interface = GripperInterface(
        node=node,
        gripper_joint_names=robot.gripper_joint_names(),
        open_gripper_joint_positions=robot.OPEN_GRIPPER_JOINT_POSITIONS,
        closed_gripper_joint_positions=robot.CLOSED_GRIPPER_JOINT_POSITIONS,
        gripper_group_name=robot.MOVE_GROUP_GRIPPER,
        callback_group=callback_gripper_group,
        gripper_command_action_name="gripper_action_controller/gripper_cmd",
    )

    # Spin the node in background thread(s) and wait a bit for initialization
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()
    node.create_rate(1.0).sleep()

    # Scale down velocity and acceleration of joints (percentage of maximum)
    moveit2.max_velocity = 0.25
    moveit2.max_acceleration = 0.25

    # Get parameters
    cartesian_jump_threshold = (
        node.get_parameter("cartesian_jump_threshold")
        .get_parameter_value()
        .double_value
    )
    cartesian_avoid_collisions = (
        node.get_parameter("cartesian_avoid_collisions")
        .get_parameter_value()
        .bool_value
    )

    # Set parameters for cartesian planning
    moveit2.cartesian_avoid_collisions = cartesian_avoid_collisions
    moveit2.cartesian_jump_threshold = cartesian_jump_threshold

    # Need to open and close gripper to initialise the controller (this is a workaround for a bug in the controller)
    init_gripper(node, gripper_interface)

    # Initialize box_poses dictionary
    box_poses = {
        'box1': {'position': None, 'orientation': None},
        'box2': {'position': None, 'orientation': None},
        'box3': {'position': None, 'orientation': None},
        'box4': {'position': None, 'orientation': None},
        'box5': {'position': None, 'orientation': None},
        'box6': {'position': None, 'orientation': None}
    }

    # Create subscriptions for each box pose
    for box_name in box_poses.keys():
        node.create_subscription(PoseStamped, f'/model/{box_name}/pose', 
                                lambda msg, name=box_name: box_detection_callback(node, msg, name, box_poses), 10)
        
    # Rotation for pointing the end effector downwards
    quat_xyzw = [1.0, 0.0, 0.0, 0.0]
    # stacking_location = [0.625, -0.1, 0.0]
    stacking_location = [0.55, 0.3, 0.1]


    # Wait for box poses to be initialized
    rclpy.spin_once(node) 
    while any(box['position'] is None for box in box_poses.values()):
        rclpy.spin_once(node)
        time.sleep(0.1) 

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()

    # Main loop: Execute pick-and-place for each box
    while rclpy.ok():
        for box_name, box_data in box_poses.items():
            if box_data['position'] is not None:
                pnp(node, moveit2, gripper_interface, box_data, quat_xyzw, stacking_location)

    rclpy.shutdown()
    executor_thread.join()
    exit(0)
