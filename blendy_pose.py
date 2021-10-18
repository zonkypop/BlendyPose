bl_info = {
    "name": "BlendyPose",
    "author": "ZonkoSoft",
    "version": (0, 1),
    "blender": (2, 80, 0),
    "location": "3D View > Sidebar > BlendyPose",
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

    # path to python.exe
    python = sys.executable

    # upgrade pip
    subprocess.call([python, "-m", "ensurepip"])
    subprocess.call([python, "-m", "pip", "install", "--upgrade", "pip"])

    # install required packages
    subprocess.call([python, "-m", "pip", "install","--target=C:\\Program Files\\Blender Foundation\\Blender 2.93\\2.93\\python\\lib", "opencv-python"])
    subprocess.call([python, "-m", "pip", "install","--target=C:\\Program Files\\Blender Foundation\\Blender 2.93\\2.93\\python\\lib", "mediapipe"])


def body_setup():
    """ Setup tracking boxes for body tracking """
    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    setup = "Pose" in scene_objects

    if not setup:
        bpy.ops.object.add(radius=1.0, type='EMPTY')
        pose = bpy.context.active_object
        pose.name = "Pose"

        bpy.ops.object.add(radius=1.0, type='EMPTY')
        body = bpy.context.active_object
        body.name = "Body"
        body.parent = pose

        for k in range(33):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = body_names[k]
            box.scale = (0.01, 0.01, 0.01)
            box.parent = body

    body = bpy.context.scene.objects["Body"]
    return body


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


def run_body(file_path):
    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        # bpy.ops.message.messagebox('INVOKE_DEFAULT', 'Installing additional libraries, this may take a moment...')
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        install()
        import cv2
        import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    body = body_setup()
    # Clean up hands and face if they were previously captured (kinda distracting)
    full_delete()

    if file_path == "None": cap = cv2.VideoCapture(0)
    else:
        print("FILE PATH: ", file_path)
        cap = cv2.VideoCapture(file_path)
        cap.open(file_path)

    with mp_pose.Pose(
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        for n in range(9000):
        # for n in range(10):
            success, image = cap.read()
            if not success: continue

            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = pose.process(image)

            if results.pose_landmarks:
                bns = [b for b in results.pose_landmarks.landmark]
                scale = 2
                bones = sorted(body.children, key=lambda b: b.name)

                for k in range(33):
                    bones[k].location.y = (bns[k].z)*0.5
                    bones[k].location.x = (0.5-bns[k].x)*scale
                    bones[k].location.z = (0.5-bns[k].y)*scale
                    bones[k].keyframe_insert(data_path="location", frame=n)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            image = cv2.flip(image, 1)

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            bpy.context.scene.frame_set(n)

        cap.release()
        cv2.destroyAllWindows()


def full_setup():
    """ Setup tracking boxes for body, face and hand tracking """
    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    pose = bpy.context.scene.objects["Pose"]

    if "Hand Left" not in scene_objects:
        bpy.ops.object.add(radius=1.0, type='EMPTY')
        hand_left = bpy.context.active_object
        hand_left.name = "Hand Left"
        hand_left.parent = pose

        for k in range(21):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(2) + "Hand Left"
            box.scale = (0.01, 0.01, 0.01)
            box.parent = hand_left

    if "Hand Right" not in scene_objects:
        bpy.ops.object.add(radius=1.0, type='EMPTY')
        hand_right = bpy.context.active_object
        hand_right.name = "Hand Right"
        hand_right.parent = pose

        for k in range(21):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(2) + "Hand Right"
            box.scale = (0.01, 0.01, 0.01)
            box.parent = hand_right

    if "Face" not in scene_objects:
        bpy.ops.object.add(radius=1.0, type='EMPTY')
        face = bpy.context.active_object
        face.name = "Face"
        face.parent = pose

        for k in range(468):
            bpy.ops.mesh.primitive_cube_add()
            box = bpy.context.active_object
            box.name = str(k).zfill(3) + "Face"
            box.scale = (0.002, 0.002, 0.002)
            box.parent = face

    hand_left = bpy.context.scene.objects["Hand Left"]
    hand_right = bpy.context.scene.objects["Hand Right"]
    face = bpy.context.scene.objects["Face"]
    return hand_left, hand_right, face


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

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    body = body_setup()
    hand_left, hand_right, face = full_setup()

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    if file_path == "None": cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_path)
        # cap.open(file_path)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        for n in range(9000):
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                bns = [b for b in results.pose_landmarks.landmark]
                scale = 2
                bones = sorted(body.children, key=lambda b: b.name)

                for k in range(33):
                    bones[k].location.y = (bns[k].z)*0.5
                    bones[k].location.x = (0.5-bns[k].x)*scale
                    bones[k].location.z = (0.5-bns[k].y)*scale
                    bones[k].keyframe_insert(data_path="location", frame=n)

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

            if results.face_landmarks:
                bns = [b for b in results.face_landmarks.landmark]
                scale = 2
                bones = sorted(face.children, key=lambda b: b.name)
                for k in range(468):
                    bones[k].location.y = (bns[k].z)*scale
                    bones[k].location.x = (0.5-bns[k].x)*scale
                    bones[k].location.z = (0.5-bns[k].y)*scale
                    bones[k].keyframe_insert(data_path="location", frame=n)


            mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

            mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

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
    bl_label = "select file"
    filename_ext = ""

    def execute(self, context):
        file_dir = self.properties.filepath
        if context.scene.settings.body_tracking: run_body(file_dir)
        else: run_full(file_dir)
        return{'FINISHED'}


class Settings(PropertyGroup):
    # Capture only body pose if True, otherwise capture hands, face and body
    body_tracking: bpy.props.BoolProperty(default=True)


class RunOperator(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.run_body_operator"
    bl_label = "Run Body Operator"

    def execute(self, context):
        if context.scene.settings.body_tracking: run_body("None")
        else: run_full("None")
        return {'FINISHED'}


class MessageBox(bpy.types.Operator):
    bl_idname = "message.messagebox"
    bl_label = ""

    message = bpy.props.StringProperty(
        name = "message",
        description = "message",
        default = 'Installing additional libraries, this may take a moment...'
    )

    def execute(self, context):
        self.report({'INFO'}, self.message)
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width = 400)

    def draw(self, context):
        self.layout.label(text=self.message)


class BlendyPosePanel(bpy.types.Panel):
    bl_label = "Blendy Pose"
    bl_category = "BlendyPose"
    bl_idname = "BlendyPose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = "Blendy Pose"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        row = layout.row()
        settings = context.scene.settings

        row = layout.row()
        row.label(text="Run Motion Capture")

        row = layout.row()
        row.operator(RunOperator.bl_idname, text="Camera")

        row = layout.row()
        row.operator(RunFileSelector.bl_idname, text="Video File")

        row = layout.row()
        row.label(text="(Press escape to stop)")

        row = layout.row()
        row.label(text="Capture Mode")

        row = layout.row()
        label = "Body" if settings.body_tracking else "Body, Hands and Face"
        row.prop(settings, 'body_tracking', text=label, toggle=True)


_classes = [
    Settings,
    BlendyPosePanel,
    RunOperator,
    RunFileSelector,
    MessageBox
]


def register():
    for c in _classes: register_class(c)
    bpy.types.Scene.settings = bpy.props.PointerProperty(type=Settings)



def unregister():
    for c in _classes: unregister_class(c)
    del bpy.types.Scene.settings


if __name__ == "__main__":
    register()
