#!/usr/bin/env python
"""
User Interface Node for PRIME

Keyboard-only interface for query/response interactions.
"""

import rospy
import sys
import select
from threading import Thread, Lock
from typing import List, Optional

try:
    from prime_ros.msg import PRIMEQuery, PRIMEResponse
    MSGS_AVAILABLE = True
except ImportError:
    rospy.logwarn("PRIME messages not built yet.")
    MSGS_AVAILABLE = False


class UserInterface:
    """Keyboard interface for PRIME interactions."""

    def __init__(self):
        rospy.init_node("user_interface", anonymous=False)

        self.query_timeout = rospy.get_param("ui/query_timeout", 30.0)

        self.lock = Lock()
        self.pending_query = None  # type: Optional[PRIMEQuery]
        self.query_start_time: Optional[rospy.Time] = None

        if MSGS_AVAILABLE:
            self.query_sub = rospy.Subscriber(
                "/prime/query",
                PRIMEQuery,
                self.query_callback,
            )

            self.response_pub = rospy.Publisher(
                "/prime/response",
                PRIMEResponse,
                queue_size=10,
            )

        self.running = True
        self.input_thread = Thread(target=self.input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

        rospy.loginfo("User Interface initialized (keyboard input)")
        rospy.loginfo("Keyboard controls:")
        rospy.loginfo("  y/1 = Yes / Option 1")
        rospy.loginfo("  n/2 = No / Option 2")
        rospy.loginfo("  3-5 = Options 3-5")
        rospy.loginfo("  q = Quit current query")

    def query_callback(self, msg):
        with self.lock:
            self.pending_query = msg
            self.query_start_time = rospy.Time.now()

        self.display_query(msg)

    def display_query(self, query):
        print("\n" + "=" * 50)
        print("PRIME QUERY")
        print("=" * 50)

        type_names = {0: "Question", 1: "Suggestion", 2: "Confirmation"}
        print("Type: %s" % type_names.get(query.query_type, "Unknown"))
        print("\n%s" % query.content)

        if query.options:
            print("\nOptions:")
            for i, opt in enumerate(query.options):
                print("  %d) %s" % (i + 1, opt))
        elif query.query_type == 2:
            print("\n  y) Yes")
            print("  n) No")

        if query.timeout > 0:
            print("\n(Timeout in %.0f seconds)" % query.timeout)

        print("=" * 50)
        print("Enter your choice: ", end="", flush=True)

    def send_response(self, selected_indices: List[int], timed_out: bool = False):
        with self.lock:
            if not self.pending_query:
                return

            query = self.pending_query

            response = PRIMEResponse()
            response.header.stamp = rospy.Time.now()
            response.query_id = query.query_id
            response.selected_indices = selected_indices
            response.timed_out = timed_out

            response.selected_labels = []
            for idx in selected_indices:
                if query.options and idx < len(query.options):
                    response.selected_labels.append(query.options[idx])
                elif idx == 0:
                    response.selected_labels.append("Yes")
                elif idx == 1:
                    response.selected_labels.append("No")

            if self.query_start_time:
                response.response_time = (rospy.Time.now() - self.query_start_time).to_sec()

            if MSGS_AVAILABLE:
                self.response_pub.publish(response)

            self.pending_query = None
            self.query_start_time = None

            rospy.loginfo("Response sent: %s", response.selected_labels)

    def input_loop(self):
        while self.running and not rospy.is_shutdown():
            try:
                with self.lock:
                    if self.pending_query and self.query_start_time:
                        elapsed = (rospy.Time.now() - self.query_start_time).to_sec()
                        if elapsed > self.pending_query.timeout and self.pending_query.timeout > 0:
                            rospy.logwarn("Query timed out")
                            self.send_response([], timed_out=True)
                            continue

                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.readline().strip().lower()

                    with self.lock:
                        if not self.pending_query:
                            continue

                        query = self.pending_query
                        selected = None

                        if key == "y" or key == "1":
                            selected = 0
                        elif key == "n" or key == "2":
                            selected = 1
                        elif key == "3":
                            selected = 2
                        elif key == "4":
                            selected = 3
                        elif key == "5":
                            selected = 4
                        elif key == "q":
                            print("\nQuery cancelled")
                            self.pending_query = None
                            continue

                        if selected is not None:
                            if query.options:
                                if selected < len(query.options):
                                    self.send_response([selected])
                                else:
                                    print("Invalid option. Choose 1-%d" % len(query.options))
                            else:
                                if selected <= 1:
                                    self.send_response([selected])
                                else:
                                    print("Invalid option. Choose y/n or 1/2")

            except Exception as exc:
                rospy.logwarn("Input error: %s", exc)

            rospy.sleep(0.1)

    def run(self):
        rospy.spin()
        self.running = False


def main():
    try:
        ui = UserInterface()
        ui.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
