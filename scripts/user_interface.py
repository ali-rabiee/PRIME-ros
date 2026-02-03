#!/usr/bin/env python
"""
User Interface Node for PRIME

Handles user queries and responses through:
1. Terminal interface (keyboard input)
2. Kinova joystick buttons (when available)

This node:
- Subscribes to PRIMEQuery messages
- Displays questions to the user
- Captures responses (keyboard or joystick buttons)
- Publishes PRIMEResponse messages
"""

import rospy
import sys
import select
import termios
import tty
from threading import Thread, Lock
from typing import Optional, List

try:
    from prime_ros.msg import (
        PRIMEQuery, PRIMEResponse, JoystickState
    )
    MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("PRIME messages not built yet.")
    MSGS_AVAILABLE = False


class UserInterface:
    """
    User interface for PRIME interactions.
    
    Supports both keyboard and joystick button input.
    """
    
    def __init__(self):
        rospy.init_node('user_interface', anonymous=False)
        
        # Parameters
        self.query_timeout = rospy.get_param('ui/query_timeout', 30.0)
        
        # Button mappings for joystick (customize based on your setup)
        self.button_yes = rospy.get_param('ui/button_yes', 0)
        self.button_no = rospy.get_param('ui/button_no', 1)
        self.button_options = [
            rospy.get_param('ui/button_option_1', 2),
            rospy.get_param('ui/button_option_2', 3),
            rospy.get_param('ui/button_option_3', 4),
            rospy.get_param('ui/button_option_4', 5),
            rospy.get_param('ui/button_option_5', 6),
        ]
        
        # State
        self.lock = Lock()
        self.pending_query: Optional[PRIMEQuery] = None
        self.query_start_time: Optional[rospy.Time] = None
        self.joystick_state: Optional[JoystickState] = None
        
        # Terminal settings for non-blocking input
        self.old_settings = None
        
        # Subscribers
        if MSGS_AVAILABLE:
            self.query_sub = rospy.Subscriber(
                '/prime/query',
                PRIMEQuery,
                self.query_callback
            )
            
            self.joystick_sub = rospy.Subscriber(
                '/prime/joystick_state',
                JoystickState,
                self.joystick_callback
            )
        
        # Publishers
        if MSGS_AVAILABLE:
            self.response_pub = rospy.Publisher(
                '/prime/response',
                PRIMEResponse,
                queue_size=10
            )
        
        # Start input thread
        self.running = True
        self.input_thread = Thread(target=self.input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
        
        rospy.loginfo("User Interface initialized")
        rospy.loginfo("Keyboard controls:")
        rospy.loginfo("  y/1 = Yes / Option 1")
        rospy.loginfo("  n/2 = No / Option 2")
        rospy.loginfo("  3-5 = Options 3-5")
        rospy.loginfo("  q = Quit current query")
    
    def query_callback(self, msg: PRIMEQuery):
        """Handle incoming queries."""
        with self.lock:
            self.pending_query = msg
            self.query_start_time = rospy.Time.now()
        
        # Display the query
        self.display_query(msg)
    
    def joystick_callback(self, msg: JoystickState):
        """Handle joystick state updates."""
        with self.lock:
            prev_state = self.joystick_state
            self.joystick_state = msg
            
            # Check for button presses (rising edge)
            if prev_state and self.pending_query:
                self.check_joystick_buttons(prev_state, msg)
    
    def check_joystick_buttons(self, prev: JoystickState, curr: JoystickState):
        """Check for joystick button presses and handle response."""
        if not self.pending_query:
            return
        
        query = self.pending_query
        
        # Check each button for rising edge (0->1)
        def button_pressed(idx):
            if idx >= len(curr.button_values) or idx >= len(prev.button_values):
                return False
            return curr.button_values[idx] == 1 and prev.button_values[idx] == 0
        
        selected_index = None
        
        # Yes/No for confirmation queries
        if query.query_type == 2:  # Confirmation
            if button_pressed(self.button_yes):
                selected_index = 0  # Yes
            elif button_pressed(self.button_no):
                selected_index = 1  # No
        
        # Multiple choice
        else:
            for i, btn_idx in enumerate(self.button_options):
                if i < len(query.options) and button_pressed(btn_idx):
                    selected_index = i
                    break
        
        if selected_index is not None:
            self.send_response([selected_index])
    
    def display_query(self, query: PRIMEQuery):
        """Display a query to the user."""
        print("\n" + "="*50)
        print("PRIME QUERY")
        print("="*50)
        
        # Query type
        type_names = {0: 'Question', 1: 'Suggestion', 2: 'Confirmation'}
        print(f"Type: {type_names.get(query.query_type, 'Unknown')}")
        
        # Content
        print(f"\n{query.content}")
        
        # Options
        if query.options:
            print("\nOptions:")
            for i, opt in enumerate(query.options):
                print(f"  {i+1}) {opt}")
        else:
            # Default Yes/No for confirmations
            if query.query_type == 2:
                print("\n  y) Yes")
                print("  n) No")
        
        # Timeout info
        if query.timeout > 0:
            print(f"\n(Timeout in {query.timeout:.0f} seconds)")
        
        print("="*50)
        print("Enter your choice: ", end='', flush=True)
    
    def send_response(self, selected_indices: List[int], timed_out: bool = False):
        """Send response to the pending query."""
        with self.lock:
            if not self.pending_query:
                return
            
            query = self.pending_query
            
            response = PRIMEResponse()
            response.header.stamp = rospy.Time.now()
            response.query_id = query.query_id
            response.selected_indices = selected_indices
            response.timed_out = timed_out
            
            # Get selected labels
            response.selected_labels = []
            for idx in selected_indices:
                if query.options and idx < len(query.options):
                    response.selected_labels.append(query.options[idx])
                elif idx == 0:
                    response.selected_labels.append("Yes")
                elif idx == 1:
                    response.selected_labels.append("No")
            
            # Compute response time
            if self.query_start_time:
                response.response_time = (rospy.Time.now() - self.query_start_time).to_sec()
            
            # Publish
            if self.response_pub:
                self.response_pub.publish(response)
            
            # Clear pending query
            self.pending_query = None
            self.query_start_time = None
            
            # Log
            rospy.loginfo(f"Response sent: {response.selected_labels}")
    
    def input_loop(self):
        """Background thread for keyboard input."""
        # Note: This is a simple implementation
        # For production, consider using curses or a proper terminal library
        
        while self.running and not rospy.is_shutdown():
            try:
                # Check for pending query timeout
                with self.lock:
                    if self.pending_query and self.query_start_time:
                        elapsed = (rospy.Time.now() - self.query_start_time).to_sec()
                        if elapsed > self.pending_query.timeout and self.pending_query.timeout > 0:
                            rospy.logwarn("Query timed out")
                            self.send_response([], timed_out=True)
                            continue
                
                # Non-blocking keyboard check
                # This uses select which works on Unix-like systems
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.readline().strip().lower()
                    
                    with self.lock:
                        if not self.pending_query:
                            continue
                        
                        query = self.pending_query
                        selected = None
                        
                        # Parse input
                        if key == 'y' or key == '1':
                            selected = 0
                        elif key == 'n' or key == '2':
                            selected = 1
                        elif key == '3':
                            selected = 2
                        elif key == '4':
                            selected = 3
                        elif key == '5':
                            selected = 4
                        elif key == 'q':
                            print("\nQuery cancelled")
                            self.pending_query = None
                            continue
                        
                        # Validate selection
                        if selected is not None:
                            if query.options:
                                if selected < len(query.options):
                                    self.send_response([selected])
                                else:
                                    print(f"Invalid option. Choose 1-{len(query.options)}")
                            else:
                                # Yes/No
                                if selected <= 1:
                                    self.send_response([selected])
                                else:
                                    print("Invalid option. Choose y/n or 1/2")
                
            except Exception as e:
                rospy.logwarn(f"Input error: {e}")
            
            rospy.sleep(0.1)
    
    def run(self):
        """Run the node."""
        rospy.spin()
        self.running = False


def main():
    try:
        ui = UserInterface()
        ui.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
