import json
import datetime
import uuid
import os

class InteractionLogger:
    """Handles logging of agent interactions to a file."""

    def __init__(self, log_file_path="logs/interaction_log.jsonl"):
        self.log_file_path = log_file_path
        # Ensure the directory exists if the path includes directories
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def start_session(self, user_goal):
        """Starts a new interaction session."""
        session_id = str(uuid.uuid4())
        print(f"Starting new session: {session_id} for goal: {user_goal}")
        return session_id, 0 # Return session_id and initial step_id

    def log_step(self, session_id, step_id, simplified_state, user_goal, action_taken):
        """Logs a single step of an interaction."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        log_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "step_id": step_id,
            "simplified_state": simplified_state,
            "user_goal": user_goal,
            "action_taken": action_taken,
            "outcome": "PENDING" # Default outcome, needs update later
        }

        try:
            with open(self.log_file_path, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n') # Newline for JSON Lines format
            # Return the next step_id
            return step_id + 1
        except IOError as e:
            print(f"Error writing to log file {self.log_file_path}: {e}")
            # Return current step_id if log failed, maybe retry?
            return step_id

    def update_outcome(self, session_id, final_outcome):
        """
        Updates the outcome for all PENDING steps within a given session.
        NOTE: This implementation reads the entire log, updates relevant entries,
        and rewrites the file. This is INEFFICIENT for large logs and should be
        improved for production (e.g., using a database or more targeted file I/O).
        """
        updated_lines = []
        made_update = False
        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r') as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    try:
                        entry = json.loads(line)
                        # Check if entry belongs to the target session and is PENDING
                        if entry.get("session_id") == session_id and entry.get("outcome") == "PENDING":
                            entry["outcome"] = final_outcome
                            updated_lines.append(json.dumps(entry) + '\n')
                            made_update = True
                        else:
                            updated_lines.append(line) # Keep line as is
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping malformed JSON line {i+1} during outcome update.")
                        updated_lines.append(line) # Keep malformed line as is
                    except Exception as e: # Catch other potential errors per line
                         print(f"Warning: Error processing line {i+1} during outcome update: {e}")
                         updated_lines.append(line) # Keep line as is

            if made_update:
                # Rewrite the file with updated lines
                with open(self.log_file_path, 'w') as f:
                    f.writelines(updated_lines)
                # print(f"Updated outcome to '{final_outcome}' for session {session_id}.")
            else:
                # print(f"No PENDING entries found for session {session_id} to update.")
                print(f"Warning: No PENDING entries found for session {session_id} to update.")

        except IOError as e:
            print(f"Error reading/writing log file {self.log_file_path} during outcome update: {e}")
        except Exception as e: # Catch broader errors during file operations
            print(f"An unexpected error occurred during outcome update: {e}")

# Example Usage (can be removed or placed in a separate test script)
# Removed the example usage block 