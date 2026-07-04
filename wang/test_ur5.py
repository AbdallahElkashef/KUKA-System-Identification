import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# 1. SETUP PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "robot_assets", "KR16_L6.xml")

def main():
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # ---------------------------------------------------------
    # 🎯 TEST YOUR ANGLES HERE (IN RADIANS)
    # Order: [Base, Shoulder, Elbow, Wrist1, Wrist2, Wrist3]
    # ---------------------------------------------------------
    test_q = np.array([0.0, -1.0, 1.0, 0.0, 0.0, 0.0])

    print("Launching Viewer... Press CTRL+C in the terminal to close.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set a nice zoomed-out camera angle
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0, 0, 1.0]

        while viewer.is_running():
            # Force the joints to stay exactly at your test angles
            data.qpos[:] = test_q
            
            # Update the 3D graphics (bypassing the physics engine completely)
            mujoco.mj_kinematics(model, data)
            
            viewer.sync()
            time.sleep(0.05)

if __name__ == "__main__":
    main()