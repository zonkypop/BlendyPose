bl_info = {
    "name": "PosePipe",
    "author": "ZonkoSoft, SpectralVectors",
    "version": (0, 4),
    "blender": (2, 80, 0),
    "location": "3D View > Sidebar > PosePipe",
    "description": "Motion capture using your camera!",
    "category": "3D View",
}


import bpy
from bpy.types import Panel, Operator, PropertyGroup, FloatProperty, PointerProperty
from bpy.utils import register_class, unregister_class
from bpy_extras.io_utils import ImportHelper


body_names = [
"00 nose",
"01 left eye (inner)",
"02 left eye",
"03 left eye (outer)",
"04 right eye (inner)",
"05 right eye",
"06 right eye (outer)",
"07 left ear",
"08 right ear",
"09 mouth (left)",
"10 mouth (right)",
"11 left shoulder",
"12 right shoulder",
"13 left elbow",
"14 right elbow",
"15 left wrist",
"16 right wrist",
"17 left pinky",
"18 right pinky",
"19 left index",
"20 right index",
"21 left thumb",
"22 right thumb",
"23 left hip",
"24 right hip",
"25 left knee",
"26 right knee",
"27 left ankle",
"28 right ankle",
"29 left heel",
"30 right heel",
"31 left foot index",
"32 right foot index",
]


def install():
    """ Install MediaPipe and dependencies behind the scenes """
    import subprocess
    import sys

    subprocess.check_call([
        sys.executable, 
        "-m", "ensurepip"])

    subprocess.check_call([
        sys.executable, 
        "-m", "pip", "install", "--upgrade", "pip"])

    subprocess.check_call([
        sys.executable, 
        "-m", "pip", "install",
        "--target=C:\\Program Files\\Blender Foundation\\Blender 2.93\\2.93\\python\\lib", 
        "opencv-python"])

    subprocess.check_call([
        sys.executable, 
        "-m", "pip", "install",
        "--target=C:\\Program Files\\Blender Foundation\\Blender 2.93\\2.93\\python\\lib", 
        "mediapipe"])


def body_setup():
    """ Setup tracking boxes for body tracking """

    for area in bpy.context.screen.areas: 
        if area.type == 'VIEW_3D':
            for space in area.spaces: 
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'OBJECT'

    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    setup = "Pose" in scene_objects

    if not setup:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        pose = bpy.context.active_object
        pose.name = "Pose"
        pose.scale = (-1,1,1)

    pose = bpy.context.scene.objects["Pose"]

    bpy.ops.object.add(radius=0.1, type='EMPTY')
    body = bpy.context.active_object
    body.name = "Body"
    body.parent = pose

    for k in range(33):
        bpy.ops.mesh.primitive_cube_add()
        box = bpy.context.active_object
        box.name = body_names[k]
        box.scale = [0.005, 0.005, 0.005]
        box.parent = body
        box.color = (0,255,0,255)

    body = bpy.context.scene.objects["Body"]
    return body


def hands_setup():
    """ Setup tracking boxes for hand tracking """

    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    setup = "Pose" in scene_objects

    if not setup:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        pose = bpy.context.active_object
        pose.name = "Pose"
        pose.scale = (-1,1,1)

    pose = bpy.context.scene.objects["Pose"]

    for area in bpy.context.screen.areas: 
        if area.type == 'VIEW_3D':
            for space in area.spaces: 
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'OBJECT'

    if "Hand Left" not in scene_objects:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        hand_left = bpy.context.active_object
        hand_left.name = "Hand Left"
        hand_left.parent = pose

        for k in range(21):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(2) + "Hand Left"
            box.scale = (0.01, 0.01, 0.01)
            box.parent = hand_left
            box.color = (0,0,255,255)

    if "Hand Right" not in scene_objects:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        hand_right = bpy.context.active_object
        hand_right.name = "Hand Right"
        hand_right.parent = pose

        for k in range(21):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(2) + "Hand Right"
            box.scale = (0.01, 0.01, 0.01)
            box.parent = hand_right
            box.color = (255,0,0,255)    

    hand_left = bpy.context.scene.objects["Hand Left"]
    hand_right = bpy.context.scene.objects["Hand Right"]
    pose.scale = (-1,1,1)
    return hand_left, hand_right


def face_setup():
    """ Setup tracking boxes for face tracking """

    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    setup = "Pose" in scene_objects

    if not setup:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        pose = bpy.context.active_object
        pose.name = "Pose"
        pose.scale = (-1,1,1)

    pose = bpy.context.scene.objects["Pose"]

    for area in bpy.context.screen.areas: 
        if area.type == 'VIEW_3D':
            for space in area.spaces: 
                if space.type == 'VIEW_3D':
                    space.shading.color_type = 'OBJECT'

    if "Face" not in scene_objects:
        bpy.ops.object.add(radius=0.1, type='EMPTY')
        face = bpy.context.active_object
        face.name = "Face"
        face.parent = pose

        for k in range(468):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(3) + "Face"
            box.scale = (0.004, 0.004, 0.004)
            box.parent = face
            box.color = (255,0,255,255)

    face = bpy.context.scene.objects["Face"]
    pose.scale = (-1,1,1)
    return face


def full_delete():
    """ Deletes all objects associated with full capture """
    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    pose = bpy.context.scene.objects["Pose"]

    if "Hand Left" in scene_objects:
        for c in  bpy.context.scene.objects["Hand Left"].children:
            bpy.data.objects[c.name].select_set(True)
            bpy.ops.object.delete()
        bpy.data.objects["Hand Left"].select_set(True)
        bpy.ops.object.delete()

    if "Hand Right" in scene_objects:
        for c in  bpy.context.scene.objects["Hand Right"].children:
            bpy.data.objects[c.name].select_set(True)
            bpy.ops.object.delete()
        bpy.data.objects["Hand Right"].select_set(True)
        bpy.ops.object.delete()

    if "Face" in scene_objects:
        for c in  bpy.context.scene.objects["Face"].children:
            bpy.data.objects[c.name].select_set(True)
            bpy.ops.object.delete()
        bpy.data.objects["Face"].select_set(True)
        bpy.ops.object.delete()


def run_full(file_path):
    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        # bpy.ops.message.messagebox('INVOKE_DEFAULT', message = 'Installing additional libraries, this may take a moment...')
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        install()
        import cv2
        import mediapipe as mp

    settings = bpy.context.scene.settings
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    if settings.body_tracking:
        body = body_setup()
    if settings.hand_tracking:
        hand_left, hand_right = hands_setup()
    if settings.face_tracking: 
        face = face_setup()

    if file_path == "None": cap = cv2.VideoCapture(settings.camera_number)
    else:
        cap = cv2.VideoCapture(file_path)

    with mp_holistic.Holistic(
        model_complexity=settings.model_complexity,
        smooth_landmarks=settings.smoothing,
        min_detection_confidence=settings.detection_confidence,
        min_tracking_confidence=settings.tracking_confidence) as holistic:

        for n in range(9000):
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if settings.body_tracking:
                if results.pose_landmarks:
                    bns = [b for b in results.pose_landmarks.landmark]
                    scale = 2
                    bones = sorted(body.children, key=lambda b: b.name)

                    for k in range(33):
                        bones[k].location.y = (bns[k].z)*0.5
                        bones[k].location.x = (0.5-bns[k].x)*scale
                        bones[k].location.z = (0.5-bns[k].y)*scale
                        bones[k].keyframe_insert(data_path="location", frame=n)
                        
            if settings.hand_tracking:
                if results.left_hand_landmarks:
                    bns = [b for b in results.left_hand_landmarks.landmark]
                    scale = 2
                    bones = sorted(hand_left.children, key=lambda b: b.name)
                    for k in range(21):
                        bones[k].location.y = (bns[k].z)*scale
                        bones[k].location.x = (0.5-bns[k].x)*scale
                        bones[k].location.z = (0.5-bns[k].y)*scale
                        bones[k].keyframe_insert(data_path="location", frame=n)

                if results.right_hand_landmarks:
                    bns = [b for b in results.right_hand_landmarks.landmark]
                    scale = 2
                    bones = sorted(hand_right.children, key=lambda b: b.name)
                    for k in range(21):
                        bones[k].location.y = (bns[k].z)*scale
                        bones[k].location.x = (0.5-bns[k].x)*scale
                        bones[k].location.z = (0.5-bns[k].y)*scale
                        bones[k].keyframe_insert(data_path="location", frame=n)

            if settings.face_tracking:
                if results.face_landmarks:
                    bns = [b for b in results.face_landmarks.landmark]
                    scale = 2
                    bones = sorted(face.children, key=lambda b: b.name)
                    for k in range(468):
                        bones[k].location.y = (bns[k].z)*scale
                        bones[k].location.x = (0.5-bns[k].x)*scale
                        bones[k].location.z = (0.5-bns[k].y)*scale
                        bones[k].keyframe_insert(data_path="location", frame=n)

            if settings.face_tracking: 
                mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(128,0,128), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),)
            
            if settings.hand_tracking:
                mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128,0,0), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),)

                mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,128), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1),)
            
            if settings.body_tracking:
                mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,128,0), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),)

            image = cv2.flip(image, 1)
            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            bpy.context.scene.frame_set(n)

    cap.release()
    cv2.destroyAllWindows()


def draw_file_opener(self, context):
    layout = self.layout
    scn = context.scene
    col = layout.column()
    row = col.row(align=True)
    row.prop(scn.settings, 'file_path', text='directory:')
    row.operator("something.identifier_selector", icon="FILE_FOLDER", text="")


class RunFileSelector(bpy.types.Operator, ImportHelper):
    bl_idname = "something.identifier_selector"
    bl_label = "Select Video File"
    filename_ext = ""

    def execute(self, context):
        file_dir = self.properties.filepath
        run_full(file_dir)
        return{'FINISHED'}


class RunOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.run_body_operator"
    bl_label = "Run Body Operator"

    def execute(self, context):
        run_full("None")
        return {'FINISHED'}


class Settings(PropertyGroup):
    # Capture only body pose if True, otherwise capture hands, face and body
    face_tracking: bpy.props.BoolProperty(default=False)
    hand_tracking: bpy.props.BoolProperty(default=False)
    body_tracking: bpy.props.BoolProperty(default=True)
    
    camera_number: bpy.props.IntProperty(default=0, 
                                        soft_min=0, 
                                        soft_max=10, 
                                        description="If you have more than one camera, you can choose here. 0 should work for most users.")
    
    tracking_confidence: bpy.props.FloatProperty(default=0.5,
                                                soft_min=0.1,
                                                soft_max=1,
                                                description="Minimum level of data necessary to track, higher numbers = higher latency.")
    
    detection_confidence: bpy.props.FloatProperty(default=0.5,
                                                soft_min=0.1,
                                                soft_max=1,
                                                description="Minimum level of data necessary to detect, higher numbers = higher latency.")
    
    smoothing: bpy.props.BoolProperty(default=True,
                                        description="If True, applies a smoothing pass to the tracked data.")
    
    model_complexity: bpy.props.IntProperty(default=1,
                                            soft_min=0,
                                            soft_max=2,
                                            description='Complexity of the tracking model, higher numbers = higher latency')


class BlendyPosePanel(bpy.types.Panel):
    bl_label = "PosePipe"
    bl_category = "PosePipe"
    bl_idname = "VIEW3D_PT_BlendyPose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = "PosePipe"

    def draw(self, context):

        settings = context.scene.settings

        layout = self.layout

        row = layout.row()
        row.label(text="Camera Motion Capture", icon='VIEW_CAMERA')
        
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text='Camera Settings:')
        column.operator(RunOperator.bl_idname, text="Start Camera", icon='CAMERA_DATA')
        split = column.split(factor=0.6)
        split.prop(settings, 'camera_number', text='Camera: ')
        split.label(text="to Exit", icon='EVENT_ESC')
        column.operator(RunFileSelector.bl_idname, text="Load Video File", icon='FILE_MOVIE')

        row = layout.row()
        row.label(text="Capture Mode", icon='FILE_SCRIPT')

        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text='Select Data to Capture:')
        column.prop(settings, 'face_tracking', text='Face', icon='MONKEY')
        column.prop(settings, 'hand_tracking', text='Hands', icon='VIEW_PAN')
        column.prop(settings, 'body_tracking', text='Body', icon='ARMATURE_DATA')

        row = layout.row()
        row.label(text="Capture Settings", icon='PREFERENCES')

        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.prop(settings, 'model_complexity', text='Model Complexity:')
        column.prop(settings, 'detection_confidence', text='Detection:')
        column.prop(settings, 'tracking_confidence', text='Tracking:')
        column.prop(settings, 'smoothing', text='Jitter Smoothing', icon='MOD_SMOOTH')
        
        
_classes = [
    Settings,
    BlendyPosePanel,
    RunOperator,
    RunFileSelector,
]


def register():
    for c in _classes: register_class(c)
    bpy.types.Scene.settings = bpy.props.PointerProperty(type=Settings)


def unregister():
    for c in _classes: unregister_class(c)
    del bpy.types.Scene.settings


if __name__ == "__main__":
    register()
