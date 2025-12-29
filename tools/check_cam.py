import pyrealsense2 as rs

def get_realsense_serials():
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("❌ 未检测到 RealSense 摄像头！请检查 USB 连接。")
        return

    print(f"✅ 检测到 {len(devices)} 个设备：")
    print("-" * 30)

    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"摄像头 #{i+1}")
        print(f"  型号: {name}")
        print(f"  序列号 (Serial Number): {serial}")
        print("-" * 30)

if __name__ == "__main__":
    get_realsense_serials()
