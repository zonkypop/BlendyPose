bl_info = {
    "name": "PosePipe",
    "author": "ZonkoSoft, SpectralVectors",
    "version": (0, 8),
    "blender": (2, 80, 0),
    "location": "3D View > Sidebar > PosePipe",
    "description": "Motion capture using your camera!",
    "category": "3D View",
}


import bpy
from bpy.types import Panel, Operator, PropertyGroup, FloatProperty, PointerProperty
from bpy.utils import register_class, unregister_class
from bpy_extras.io_utils import ImportHelper
import time

bone_translate = {
    'clavicle_l' : {
        'bone_name': 'clavicle_l',
        'rigify_name': 'shoulder.L',
        'unreal_name': 'clavicle_l',
        'copy_location': '11 right shoulder',
        'stretch_to': '12 left shoulder',
    }
}

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

    string = bpy.app.version_string
    blenderversion = string.rstrip(string[-2:])
    
    subprocess.check_call([
        sys.executable, 
        "-m", "ensurepip"])

    subprocess.check_call([
        sys.executable, 
        "-m", "pip", "install", "--upgrade", "pip"])

    subprocess.check_call([
        sys.executable, 
        "-m", "pip", "install",
        f"--target=C:\\Program Files\\Blender Foundation\\Blender {blenderversion}\\{blenderversion}\\python\\lib",
        "opencv-python"])

    subprocess.check_call([
        sys.executable, 
        "-m", "pip", "install",
        f"--target=C:\\Program Files\\Blender Foundation\\Blender {blenderversion}\\{blenderversion}\\python\\lib", 
        "mediapipe"])

    subprocess.check_call([
        sys.executable, 
        "-m", "pip", "install",
        f"--target=C:\\Program Files\\Blender Foundation\\Blender {blenderversion}\\{blenderversion}\\python\\lib", 
        "protobuf==3.19.0", 
        "--upgrade"])

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
        box.scale = [0.003, 0.003, 0.003]
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
            box.scale = (0.005, 0.005, 0.005)
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
            box.scale = (0.005, 0.005, 0.005)
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
            box.scale = (0.002, 0.002, 0.002)
            box.parent = face
            box.color = (255,0,255,255)

    face = bpy.context.scene.objects["Face"]
    pose.scale = (-1,1,1)
    return face

def body_delete():
    """ Deletes all objects associated with body capture """
    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    pose = bpy.context.scene.objects["Pose"]

    if "Body" in scene_objects:
        for c in bpy.context.scene.objects["Body"].children: 
            if not len(bpy.context.scene.objects["Body"].children) == 0:
                bpy.data.objects[c.name].select_set(True)
                bpy.ops.object.delete()
        bpy.data.objects["Body"].select_set(True)
        bpy.ops.object.delete()

def face_delete():
    """ Deletes all objects associated with face capture """
    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    pose = bpy.context.scene.objects["Pose"]

    if "Face" in scene_objects:
        for c in  bpy.context.scene.objects["Face"].children:
            if not len(bpy.context.scene.objects["Face"].children) == 0:
                bpy.data.objects[c.name].select_set(True)
                bpy.ops.object.delete()
        bpy.data.objects["Face"].select_set(True)
        bpy.ops.object.delete()

def hands_delete():
    """ Deletes all objects associated with hands capture """
    scene_objects = [n for n in bpy.context.scene.objects.keys()]
    pose = bpy.context.scene.objects["Pose"]
    if "Hand Left" in scene_objects:
        for c in  bpy.context.scene.objects["Hand Left"].children:
            if not len(bpy.context.scene.objects["Hand Left"].children) == 0:
                bpy.data.objects[c.name].select_set(True)
                bpy.ops.object.delete()
        bpy.data.objects["Hand Left"].select_set(True)
        bpy.ops.object.delete()

    if "Hand Right" in scene_objects:
        for c in  bpy.context.scene.objects["Hand Right"].children:
            if not len(bpy.context.scene.objects["Hand Right"].children) == 0:
                bpy.data.objects[c.name].select_set(True)
                bpy.ops.object.delete()
        bpy.data.objects["Hand Right"].select_set(True)
        bpy.ops.object.delete()

def run_full(file_path):
    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        install()
        import cv2
        import mediapipe as mp

    settings = bpy.context.scene.settings
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    if settings.body_tracking:
        #if "Body" in bpy.context.scene.objects.keys():
        #    body_delete()
        body = body_setup()
    if settings.hand_tracking:
        #if "Left Hand" or "Right Hand" in bpy.context.scene.objects.keys():
        #    hands_delete()
        hand_left, hand_right = hands_setup()
    if settings.face_tracking: 
        #if "Face" in bpy.context.scene.objects.keys():
        #    face_delete()
        face = face_setup()

    if file_path == "None": cap = cv2.VideoCapture(settings.camera_number)
    else:
        cap = cv2.VideoCapture(file_path)

    with mp_holistic.Holistic(
        model_complexity=settings.model_complexity,
        smooth_landmarks=settings.smoothing,
        min_detection_confidence=settings.detection_confidence,
        min_tracking_confidence=settings.tracking_confidence) as holistic:

        previousTime = 0

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
                        bones[k].location.y = bns[k].z / 4
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.2-bns[k].y) + 2
                        bones[k].keyframe_insert(data_path="location", frame=n)
                        
            if settings.hand_tracking:
                if results.left_hand_landmarks:
                    bns = [b for b in results.left_hand_landmarks.landmark]
                    scale = 2
                    bones = sorted(hand_left.children, key=lambda b: b.name)
                    for k in range(21):
                        bones[k].location.y = bns[k].z
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.5-bns[k].y)/2 + 1.6
                        bones[k].keyframe_insert(data_path="location", frame=n)


                if results.right_hand_landmarks:
                    bns = [b for b in results.right_hand_landmarks.landmark]
                    scale = 2
                    bones = sorted(hand_right.children, key=lambda b: b.name)
                    for k in range(21):
                        bones[k].location.y = bns[k].z
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.5-bns[k].y)/2 + 1.6
                        bones[k].keyframe_insert(data_path="location", frame=n)


            if settings.face_tracking:
                if results.face_landmarks:
                    bns = [b for b in results.face_landmarks.landmark]
                    scale = 2
                    bones = sorted(face.children, key=lambda b: b.name)
                    for k in range(468):
                        bones[k].location.y = bns[k].z
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.2-bns[k].y) + 2
                        bones[k].keyframe_insert(data_path="location", frame=n)


            if settings.face_tracking: 
                mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(128,0,128), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),)
            
            if settings.hand_tracking:
                mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128,0,0), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=3, circle_radius=1),)

                mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,128), thickness=1, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=1),)
            
            if settings.body_tracking:
                mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,128,0), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1),)

            image = cv2.flip(image, 1)
            
            currentTime = time.time()
            capture_fps = 1 / (currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(img=image, 
                        text='FPS: ' + str(int(capture_fps)), 
                        org=(10,30), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=2, 
                        color=(255,255,255), 
                        thickness=2)
            
            cv2.imshow('MediaPipe Holistic', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            bpy.context.scene.frame_set(n)

    cap.release()
    cv2.destroyAllWindows()

    settings.capture_fps = capture_fps

    # Attach hands and face to body
    if settings.face_tracking:
        bpy.context.view_layer.objects.active = bpy.data.objects['Face']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bpy.data.objects['Face'].constraints['Copy Location'].target = bpy.data.objects['Pose']
        bpy.data.objects['Face'].constraints["Copy Location"].use_y = False
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bpy.data.objects['Face'].constraints['Copy Location.001'].target = bpy.data.objects['00 nose']
        bpy.data.objects['Face'].constraints["Copy Location.001"].use_x = False
        bpy.data.objects['Face'].constraints["Copy Location.001"].use_z = False

    if settings.hand_tracking:
        bpy.context.view_layer.objects.active = bpy.data.objects['Hand Right']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bpy.data.objects['Hand Right'].constraints['Copy Location'].target = bpy.data.objects['Pose']
        bpy.data.objects['Hand Right'].constraints["Copy Location"].use_y = False
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bpy.data.objects['Hand Right'].constraints['Copy Location.001'].target = bpy.data.objects['16 right wrist']
        bpy.data.objects['Hand Right'].constraints["Copy Location.001"].use_x = False
        bpy.data.objects['Hand Right'].constraints["Copy Location.001"].use_z = False 
        
        bpy.context.view_layer.objects.active = bpy.data.objects['Hand Left']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bpy.data.objects['Hand Left'].constraints['Copy Location'].target = bpy.data.objects['Pose']
        bpy.data.objects['Hand Left'].constraints["Copy Location"].use_y = False
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bpy.data.objects['Hand Left'].constraints['Copy Location.001'].target = bpy.data.objects['15 left wrist']
        bpy.data.objects['Hand Left'].constraints["Copy Location.001"].use_x = False
        bpy.data.objects['Hand Left'].constraints["Copy Location.001"].use_z = False


class RetimeAnimation(bpy.types.Operator):
    """Builds an armature to use with the mocap data"""
    bl_idname = "posepipe.retime_animation"
    bl_label = "Retime Animation"

    def execute(self, context):

        # Retime animation
        #bpy.data.objects['Pose'].select_set(True)
        scene_objects = [n for n in bpy.context.scene.objects.keys()]
        
        if "Body" in scene_objects:
            for c in bpy.context.scene.objects["Body"].children:
                bpy.data.objects[c.name].select_set(True)
        if "Hand Left" in scene_objects:
            for c in bpy.context.scene.objects["Hand Left"].children:
                bpy.data.objects[c.name].select_set(True)
        if "Hand Right" in scene_objects:
            for c in bpy.context.scene.objects["Hand Right"].children:
                bpy.data.objects[c.name].select_set(True)
        if "Face" in scene_objects:
            for c in bpy.context.scene.objects["Face"].children:
                bpy.data.objects[c.name].select_set(True)

        bpy.data.scenes['Scene'].frame_current = 0
        frame_rate = bpy.data.scenes['Scene'].render.fps
        timescale = frame_rate / bpy.context.scene.settings.capture_fps
        #bpy.context.area.type =  bpy.data.screens['Layout'].areas[2].type
        context.area.type = 'DOPESHEET_EDITOR'
        context.area.spaces[0].mode = 'TIMELINE'
        bpy.ops.transform.transform(mode='TIME_SCALE', value=(timescale, 0, 0, 0))
        #bpy.context.area.type = bpy.data.screens['Layout'].areas[-1].type
        context.area.type = 'VIEW_3D'
        return{'FINISHED'}



def draw_file_opener(self, context):
    layout = self.layout
    scn = context.scene
    col = layout.column()
    row = col.row(align=True)
    row.prop(scn.settings, 'file_path', text='directory:')
    row.operator("something.identifier_selector", icon="FILE_FOLDER", text="")


class RunFileSelector(Operator, ImportHelper):
    bl_idname = "something.identifier_selector"
    bl_label = "Select Video File"
    filename_ext = ""

    def execute(self, context):
        file_dir = self.properties.filepath
        run_full(file_dir)
        return{'FINISHED'}


class RunOperator(Operator):
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

    capture_fps: bpy.props.IntProperty(default=0,
                                        description='Framerate of the motion capture')

class SkeletonBuilder(bpy.types.Operator):
    """Builds an armature to use with the mocap data"""
    bl_idname = "pose.skeleton_builder"
    bl_label = "Skeleton Builder"

    def execute(self, context):

        settings = bpy.context.scene.settings

        bpy.ops.object.armature_add(radius=0.1)

        PosePipe_BodyBones = bpy.context.object
        PosePipe_BodyBones.name = "PosePipe_BodyBones"

        bpy.data.armatures['Armature'].name = "Body_Skeleton"
        Body_Skeleton = bpy.data.armatures["Body_Skeleton"]
        Body_Skeleton.display_type = 'STICK'

        bpy.data.armatures["Body_Skeleton"].bones["Bone"].name = "root"

        bpy.ops.object.editmode_toggle()

        root = bpy.context.active_object.data.edit_bones["root"]

        bpy.ops.armature.bone_primitive_add(name="pelvis")
        pelvis = bpy.context.active_object.data.edit_bones["pelvis"]
        bpy.context.active_object.data.edit_bones["pelvis"].tail[2] = 0.1
        pelvis.parent = root

        bpy.ops.armature.bone_primitive_add(name="spine01")
        spine01 = bpy.context.active_object.data.edit_bones["spine01"]
        bpy.context.active_object.data.edit_bones["spine01"].tail[2] = 0.1
        spine01.parent = pelvis

        bpy.ops.armature.bone_primitive_add(name="spine02")
        spine02 = bpy.context.active_object.data.edit_bones["spine02"]
        bpy.context.active_object.data.edit_bones["spine02"].tail[2] = 0.1
        spine02.parent = spine01

        bpy.ops.armature.bone_primitive_add(name="spine03")
        spine03 = bpy.context.active_object.data.edit_bones["spine03"]
        bpy.context.active_object.data.edit_bones["spine03"].tail[2] = 0.1
        spine03.parent = spine02

        bpy.ops.armature.bone_primitive_add(name="neck_01")
        neck_01 = bpy.context.active_object.data.edit_bones["neck_01"]
        bpy.context.active_object.data.edit_bones["neck_01"].tail[2] = 0.1
        neck_01.parent = spine03

        bpy.ops.armature.bone_primitive_add(name="head")
        head = bpy.context.active_object.data.edit_bones["head"]
        bpy.context.active_object.data.edit_bones["head"].tail[2] = 0.1
        head.parent = neck_01

        bpy.ops.armature.bone_primitive_add(name="thigh_l")
        thigh_l = bpy.context.active_object.data.edit_bones["thigh_l"]
        bpy.context.active_object.data.edit_bones["thigh_l"].tail[2] = 0.1
        thigh_l.parent = pelvis

        bpy.ops.armature.bone_primitive_add(name="calf_l")
        calf_l = bpy.context.active_object.data.edit_bones["calf_l"]
        bpy.context.active_object.data.edit_bones["calf_l"].tail[2] = 0.1
        calf_l.parent = thigh_l

        bpy.ops.armature.bone_primitive_add(name="foot_l")
        foot_l = bpy.context.active_object.data.edit_bones["foot_l"]
        bpy.context.active_object.data.edit_bones["foot_l"].tail[2] = 0.1
        foot_l.parent = calf_l

        bpy.ops.armature.bone_primitive_add(name="thigh_r")
        thigh_r = bpy.context.active_object.data.edit_bones["thigh_r"]
        bpy.context.active_object.data.edit_bones["thigh_r"].tail[2] = 0.1
        thigh_r.parent = pelvis

        bpy.ops.armature.bone_primitive_add(name="calf_r")
        calf_r = bpy.context.active_object.data.edit_bones["calf_r"]
        bpy.context.active_object.data.edit_bones["calf_r"].tail[2] = 0.1
        calf_r.parent = thigh_r

        bpy.ops.armature.bone_primitive_add(name="foot_r")
        foot_r = bpy.context.active_object.data.edit_bones["foot_r"]
        bpy.context.active_object.data.edit_bones["foot_r"].tail[2] = 0.1
        foot_r.parent = calf_r

        bpy.ops.armature.bone_primitive_add(name="clavicle_l")
        clavicle_l = bpy.context.active_object.data.edit_bones["clavicle_l"]
        bpy.context.active_object.data.edit_bones["clavicle_l"].tail[2] = 0.1
        clavicle_l.parent = spine03

        bpy.ops.armature.bone_primitive_add(name="upperarm_l")
        upperarm_l = bpy.context.active_object.data.edit_bones["upperarm_l"]
        bpy.context.active_object.data.edit_bones["upperarm_l"].tail[2] = 0.1
        upperarm_l.parent = clavicle_l

        bpy.ops.armature.bone_primitive_add(name="lowerarm_l")
        lowerarm_l = bpy.context.active_object.data.edit_bones["lowerarm_l"]
        bpy.context.active_object.data.edit_bones["lowerarm_l"].tail[2] = 0.1
        lowerarm_l.parent = upperarm_l

        bpy.ops.armature.bone_primitive_add(name="clavicle_r")
        clavicle_r = bpy.context.active_object.data.edit_bones["clavicle_r"]
        bpy.context.active_object.data.edit_bones["clavicle_r"].tail[2] = 0.1
        clavicle_r.parent = spine03

        bpy.ops.armature.bone_primitive_add(name="upperarm_r")
        upperarm_r = bpy.context.active_object.data.edit_bones["upperarm_r"]
        bpy.context.active_object.data.edit_bones["upperarm_r"].tail[2] = 0.1
        upperarm_r.parent = clavicle_r

        bpy.ops.armature.bone_primitive_add(name="lowerarm_r")
        lowerarm_r = bpy.context.active_object.data.edit_bones["lowerarm_r"]
        bpy.context.active_object.data.edit_bones["lowerarm_r"].tail[2] = 0.1
        lowerarm_r.parent = upperarm_r
        
        if settings.hand_tracking:
            bpy.ops.armature.bone_primitive_add(name="hand_l")
            hand_l = bpy.context.active_object.data.edit_bones["hand_l"]
            bpy.context.active_object.data.edit_bones["hand_l"].tail[2] = 0.1
            hand_l.parent = lowerarm_l

            bpy.ops.armature.bone_primitive_add(name="thumb_01_l")
            thumb_01_l = bpy.context.active_object.data.edit_bones["thumb_01_l"]
            bpy.context.active_object.data.edit_bones["thumb_01_l"].tail[2] = 0.1
            thumb_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="thumb_02_l")
            thumb_02_l = bpy.context.active_object.data.edit_bones["thumb_02_l"]
            bpy.context.active_object.data.edit_bones["thumb_02_l"].tail[2] = 0.1
            thumb_02_l.parent = thumb_01_l

            bpy.ops.armature.bone_primitive_add(name="thumb_03_l")
            thumb_03_l = bpy.context.active_object.data.edit_bones["thumb_03_l"]
            bpy.context.active_object.data.edit_bones["thumb_03_l"].tail[2] = 0.1
            thumb_03_l.parent = thumb_02_l

            bpy.ops.armature.bone_primitive_add(name="index_01_l")
            index_01_l = bpy.context.active_object.data.edit_bones["index_01_l"]
            bpy.context.active_object.data.edit_bones["index_01_l"].tail[2] = 0.1
            index_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="index_02_l")
            index_02_l = bpy.context.active_object.data.edit_bones["index_02_l"]
            bpy.context.active_object.data.edit_bones["index_02_l"].tail[2] = 0.1
            index_02_l.parent = index_01_l

            bpy.ops.armature.bone_primitive_add(name="index_03_l")
            index_03_l = bpy.context.active_object.data.edit_bones["index_03_l"]
            bpy.context.active_object.data.edit_bones["index_03_l"].tail[2] = 0.1
            index_03_l.parent = index_02_l

            bpy.ops.armature.bone_primitive_add(name="middle_01_l")
            middle_01_l = bpy.context.active_object.data.edit_bones["middle_01_l"]
            bpy.context.active_object.data.edit_bones["middle_01_l"].tail[2] = 0.1
            middle_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="middle_02_l")
            middle_02_l = bpy.context.active_object.data.edit_bones["middle_02_l"]
            bpy.context.active_object.data.edit_bones["middle_02_l"].tail[2] = 0.1
            middle_02_l.parent = middle_01_l

            bpy.ops.armature.bone_primitive_add(name="middle_03_l")
            middle_03_l = bpy.context.active_object.data.edit_bones["middle_03_l"]
            bpy.context.active_object.data.edit_bones["middle_03_l"].tail[2] = 0.1
            middle_03_l.parent = middle_02_l

            bpy.ops.armature.bone_primitive_add(name="ring_01_l")
            ring_01_l = bpy.context.active_object.data.edit_bones["ring_01_l"]
            bpy.context.active_object.data.edit_bones["ring_01_l"].tail[2] = 0.1
            ring_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="ring_02_l")
            ring_02_l = bpy.context.active_object.data.edit_bones["ring_02_l"]
            bpy.context.active_object.data.edit_bones["ring_02_l"].tail[2] = 0.1
            ring_02_l.parent = ring_01_l

            bpy.ops.armature.bone_primitive_add(name="ring_03_l")
            ring_03_l = bpy.context.active_object.data.edit_bones["ring_03_l"]
            bpy.context.active_object.data.edit_bones["ring_03_l"].tail[2] = 0.1
            ring_03_l.parent = ring_02_l

            bpy.ops.armature.bone_primitive_add(name="pinky_01_l")
            pinky_01_l = bpy.context.active_object.data.edit_bones["pinky_01_l"]
            bpy.context.active_object.data.edit_bones["pinky_01_l"].tail[2] = 0.1
            pinky_01_l.parent = hand_l

            bpy.ops.armature.bone_primitive_add(name="pinky_02_l")
            pinky_02_l = bpy.context.active_object.data.edit_bones["pinky_02_l"]
            bpy.context.active_object.data.edit_bones["pinky_02_l"].tail[2] = 0.1
            pinky_02_l.parent = pinky_01_l

            bpy.ops.armature.bone_primitive_add(name="pinky_03_l")
            pinky_03_l = bpy.context.active_object.data.edit_bones["pinky_03_l"]
            bpy.context.active_object.data.edit_bones["pinky_03_l"].tail[2] = 0.1
            pinky_03_l.parent = pinky_02_l

            bpy.ops.armature.bone_primitive_add(name="hand_r")
            hand_r = bpy.context.active_object.data.edit_bones["hand_r"]
            bpy.context.active_object.data.edit_bones["hand_r"].tail[2] = 0.1
            hand_r.parent = lowerarm_r

            bpy.ops.armature.bone_primitive_add(name="thumb_01_r")
            thumb_01_r = bpy.context.active_object.data.edit_bones["thumb_01_r"]
            bpy.context.active_object.data.edit_bones["thumb_01_r"].tail[2] = 0.1
            thumb_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="thumb_02_r")
            thumb_02_r = bpy.context.active_object.data.edit_bones["thumb_02_r"]
            bpy.context.active_object.data.edit_bones["thumb_02_r"].tail[2] = 0.1
            thumb_02_r.parent = thumb_01_r

            bpy.ops.armature.bone_primitive_add(name="thumb_03_r")
            thumb_03_r = bpy.context.active_object.data.edit_bones["thumb_03_r"]
            bpy.context.active_object.data.edit_bones["thumb_03_r"].tail[2] = 0.1
            thumb_03_r.parent = thumb_02_r

            bpy.ops.armature.bone_primitive_add(name="index_01_r")
            index_01_r = bpy.context.active_object.data.edit_bones["index_01_r"]
            bpy.context.active_object.data.edit_bones["index_01_r"].tail[2] = 0.1
            index_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="index_02_r")
            index_02_r = bpy.context.active_object.data.edit_bones["index_02_r"]
            bpy.context.active_object.data.edit_bones["index_02_r"].tail[2] = 0.1
            index_02_r.parent = index_01_r

            bpy.ops.armature.bone_primitive_add(name="index_03_r")
            index_03_r = bpy.context.active_object.data.edit_bones["index_03_r"]
            bpy.context.active_object.data.edit_bones["index_03_r"].tail[2] = 0.1
            index_03_r.parent = index_02_r

            bpy.ops.armature.bone_primitive_add(name="middle_01_r")
            middle_01_r = bpy.context.active_object.data.edit_bones["middle_01_r"]
            bpy.context.active_object.data.edit_bones["middle_01_r"].tail[2] = 0.1
            middle_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="middle_02_r")
            middle_02_r = bpy.context.active_object.data.edit_bones["middle_02_r"]
            bpy.context.active_object.data.edit_bones["middle_02_r"].tail[2] = 0.1
            middle_02_r.parent = middle_01_r

            bpy.ops.armature.bone_primitive_add(name="middle_03_r")
            middle_03_r = bpy.context.active_object.data.edit_bones["middle_03_r"]
            bpy.context.active_object.data.edit_bones["middle_03_r"].tail[2] = 0.1
            middle_03_r.parent = middle_02_r

            bpy.ops.armature.bone_primitive_add(name="ring_01_r")
            ring_01_r = bpy.context.active_object.data.edit_bones["ring_01_r"]
            bpy.context.active_object.data.edit_bones["ring_01_r"].tail[2] = 0.1
            ring_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="ring_02_r")
            ring_02_r = bpy.context.active_object.data.edit_bones["ring_02_r"]
            bpy.context.active_object.data.edit_bones["ring_02_r"].tail[2] = 0.1
            ring_02_r.parent = ring_01_r

            bpy.ops.armature.bone_primitive_add(name="ring_03_r")
            ring_03_r = bpy.context.active_object.data.edit_bones["ring_03_r"]
            bpy.context.active_object.data.edit_bones["ring_03_r"].tail[2] = 0.1
            ring_03_r.parent = ring_02_r

            bpy.ops.armature.bone_primitive_add(name="pinky_01_r")
            pinky_01_r = bpy.context.active_object.data.edit_bones["pinky_01_r"]
            bpy.context.active_object.data.edit_bones["pinky_01_r"].tail[2] = 0.1
            pinky_01_r.parent = hand_r

            bpy.ops.armature.bone_primitive_add(name="pinky_02_r")
            pinky_02_r = bpy.context.active_object.data.edit_bones["pinky_02_r"]
            bpy.context.active_object.data.edit_bones["pinky_02_r"].tail[2] = 0.1
            pinky_02_r.parent = pinky_01_r

            bpy.ops.armature.bone_primitive_add(name="pinky_03_r")
            pinky_03_r = bpy.context.active_object.data.edit_bones["pinky_03_r"]
            bpy.context.active_object.data.edit_bones["pinky_03_r"].tail[2] = 0.1
            pinky_03_r.parent = pinky_02_r

        bpy.ops.object.posemode_toggle()

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pelvis'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["pelvis"].constraints["Copy Location"].target = bpy.data.objects["23 left hip"]
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["pelvis"].constraints["Copy Location.001"].target = bpy.data.objects["24 right hip"]
        bpy.context.object.pose.bones["pelvis"].constraints["Copy Location.001"].influence = 0.5

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['spine01'].bone
        #bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        PosePipe_BodyBones.pose.bones['spine01'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['spine02'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['spine03'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['neck_01'].location[1] = 0.1
        PosePipe_BodyBones.pose.bones['head'].location[1] = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['spine03'].bone
        bpy.ops.pose.constraint_add(type='IK')
        bpy.context.object.pose.bones["spine03"].constraints["IK"].target = PosePipe_BodyBones
        bpy.context.object.pose.bones["spine03"].constraints["IK"].subtarget = 'neck_01'
        bpy.context.object.pose.bones["spine03"].constraints["IK"].chain_count = 3
        bpy.context.object.pose.bones["spine03"].constraints["IK"].pole_target = PosePipe_BodyBones
        bpy.context.object.pose.bones["spine03"].constraints["IK"].pole_subtarget = 'neck_01'

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['clavicle_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["clavicle_l"].constraints['Copy Location'].target = bpy.data.objects['11 left shoulder']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["clavicle_l"].constraints['Stretch To'].target = bpy.data.objects['12 right shoulder']
        bpy.context.object.pose.bones["clavicle_l"].constraints['Stretch To'].rest_length = 0.4
        bpy.context.object.pose.bones["clavicle_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["clavicle_l"].constraints['Stretch To'].keep_axis = 'PLANE_Z'

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['upperarm_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["upperarm_l"].constraints['Copy Location'].target = bpy.data.objects['11 left shoulder']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["upperarm_l"].constraints['Stretch To'].target = bpy.data.objects['13 left elbow']
        bpy.context.object.pose.bones["upperarm_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["upperarm_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['lowerarm_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["lowerarm_l"].constraints['Copy Location'].target = bpy.data.objects['13 left elbow']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["lowerarm_l"].constraints['Stretch To'].target = bpy.data.objects['15 left wrist']
        bpy.context.object.pose.bones["lowerarm_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["lowerarm_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['clavicle_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["clavicle_r"].constraints['Copy Location'].target = bpy.data.objects['12 right shoulder']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["clavicle_r"].constraints['Stretch To'].target = bpy.data.objects['11 left shoulder']
        bpy.context.object.pose.bones["clavicle_r"].constraints['Stretch To'].rest_length = 0.4
        bpy.context.object.pose.bones["clavicle_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["clavicle_r"].constraints['Stretch To'].keep_axis = 'PLANE_Z'

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['upperarm_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["upperarm_r"].constraints['Copy Location'].target = bpy.data.objects['12 right shoulder']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["upperarm_r"].constraints['Stretch To'].target = bpy.data.objects['14 right elbow']
        bpy.context.object.pose.bones["upperarm_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["upperarm_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['lowerarm_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["lowerarm_r"].constraints['Copy Location'].target = bpy.data.objects['14 right elbow']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["lowerarm_r"].constraints['Stretch To'].target = bpy.data.objects['16 right wrist']
        bpy.context.object.pose.bones["lowerarm_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["lowerarm_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thigh_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["thigh_l"].constraints['Copy Location'].target = bpy.data.objects['23 left hip']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["thigh_l"].constraints['Stretch To'].target = bpy.data.objects['25 left knee']
        bpy.context.object.pose.bones["thigh_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["thigh_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['calf_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["calf_l"].constraints['Copy Location'].target = bpy.data.objects['25 left knee']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["calf_l"].constraints['Stretch To'].target = bpy.data.objects['27 left ankle']
        bpy.context.object.pose.bones["calf_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["calf_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['foot_l'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["foot_l"].constraints['Copy Location'].target = bpy.data.objects['27 left ankle']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["foot_l"].constraints['Stretch To'].target = bpy.data.objects['31 left foot index']
        bpy.context.object.pose.bones["foot_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["foot_l"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thigh_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["thigh_r"].constraints['Copy Location'].target = bpy.data.objects['24 right hip']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["thigh_r"].constraints['Stretch To'].target = bpy.data.objects['26 right knee']
        bpy.context.object.pose.bones["thigh_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["thigh_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['calf_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["calf_r"].constraints['Copy Location'].target = bpy.data.objects['26 right knee']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["calf_r"].constraints['Stretch To'].target = bpy.data.objects['28 right ankle']
        bpy.context.object.pose.bones["calf_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["calf_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['foot_r'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["foot_r"].constraints['Copy Location'].target = bpy.data.objects['28 right ankle']
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bpy.context.object.pose.bones["foot_r"].constraints['Stretch To'].target = bpy.data.objects['32 right foot index']
        bpy.context.object.pose.bones["foot_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.context.object.pose.bones["foot_r"].constraints['Stretch To'].rest_length = 0.1

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['neck_01'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["neck_01"].constraints['Copy Location'].target = bpy.data.objects['11 left shoulder']
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["neck_01"].constraints["Copy Location.001"].target = bpy.data.objects["12 right shoulder"]
        bpy.context.object.pose.bones["neck_01"].constraints["Copy Location.001"].influence = 0.5

        bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['head'].bone
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["head"].constraints['Copy Location'].target = bpy.data.objects['09 mouth (left)']
        bpy.context.object.pose.bones["head"].constraints['Copy Location'].use_y = False
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["head"].constraints["Copy Location.001"].target = bpy.data.objects["10 mouth (right)"]
        bpy.context.object.pose.bones["head"].constraints["Copy Location.001"].influence = 0.5
        bpy.context.object.pose.bones["head"].constraints["Copy Location.001"].use_y = False
        bpy.ops.pose.constraint_add(type='COPY_LOCATION')
        bpy.context.object.pose.bones["head"].constraints["Copy Location.002"].target = bpy.data.objects["08 right ear"]
        bpy.context.object.pose.bones["head"].constraints["Copy Location.002"].use_x = False
        bpy.context.object.pose.bones["head"].constraints["Copy Location.002"].use_z = False

        if settings.hand_tracking:
            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['hand_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["hand_r"].constraints['Copy Location'].target = bpy.data.objects['00Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["hand_r"].constraints['Stretch To'].target = bpy.data.objects['09Hand Right']
            bpy.context.object.pose.bones["hand_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["hand_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["thumb_01_r"].constraints['Copy Location'].target = bpy.data.objects['01Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["thumb_01_r"].constraints['Stretch To'].target = bpy.data.objects['02Hand Right']
            bpy.context.object.pose.bones["thumb_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["thumb_02_r"].constraints['Copy Location'].target = bpy.data.objects['02Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["thumb_02_r"].constraints['Stretch To'].target = bpy.data.objects['03Hand Right']
            bpy.context.object.pose.bones["thumb_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["thumb_03_r"].constraints['Copy Location'].target = bpy.data.objects['03Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["thumb_03_r"].constraints['Stretch To'].target = bpy.data.objects['04Hand Right']
            bpy.context.object.pose.bones["thumb_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["index_01_r"].constraints['Copy Location'].target = bpy.data.objects['05Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["index_01_r"].constraints['Stretch To'].target = bpy.data.objects['06Hand Right']
            bpy.context.object.pose.bones["index_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["index_02_r"].constraints['Copy Location'].target = bpy.data.objects['06Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["index_02_r"].constraints['Stretch To'].target = bpy.data.objects['07Hand Right']
            bpy.context.object.pose.bones["index_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["index_03_r"].constraints['Copy Location'].target = bpy.data.objects['07Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["index_03_r"].constraints['Stretch To'].target = bpy.data.objects['08Hand Right']
            bpy.context.object.pose.bones["index_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["middle_01_r"].constraints['Copy Location'].target = bpy.data.objects['09Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["middle_01_r"].constraints['Stretch To'].target = bpy.data.objects['10Hand Right']
            bpy.context.object.pose.bones["middle_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["middle_02_r"].constraints['Copy Location'].target = bpy.data.objects['10Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["middle_02_r"].constraints['Stretch To'].target = bpy.data.objects['11Hand Right']
            bpy.context.object.pose.bones["middle_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["middle_03_r"].constraints['Copy Location'].target = bpy.data.objects['11Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["middle_03_r"].constraints['Stretch To'].target = bpy.data.objects['12Hand Right']
            bpy.context.object.pose.bones["middle_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["ring_01_r"].constraints['Copy Location'].target = bpy.data.objects['13Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["ring_01_r"].constraints['Stretch To'].target = bpy.data.objects['14Hand Right']
            bpy.context.object.pose.bones["ring_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["ring_02_r"].constraints['Copy Location'].target = bpy.data.objects['14Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["ring_02_r"].constraints['Stretch To'].target = bpy.data.objects['15Hand Right']
            bpy.context.object.pose.bones["ring_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["ring_03_r"].constraints['Copy Location'].target = bpy.data.objects['15Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["ring_03_r"].constraints['Stretch To'].target = bpy.data.objects['16Hand Right']
            bpy.context.object.pose.bones["ring_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_01_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["pinky_01_r"].constraints['Copy Location'].target = bpy.data.objects['17Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["pinky_01_r"].constraints['Stretch To'].target = bpy.data.objects['18Hand Right']
            bpy.context.object.pose.bones["pinky_01_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_01_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_02_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["pinky_02_r"].constraints['Copy Location'].target = bpy.data.objects['18Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["pinky_02_r"].constraints['Stretch To'].target = bpy.data.objects['19Hand Right']
            bpy.context.object.pose.bones["pinky_02_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_02_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_03_r'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["pinky_03_r"].constraints['Copy Location'].target = bpy.data.objects['19Hand Right']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["pinky_03_r"].constraints['Stretch To'].target = bpy.data.objects['20Hand Right']
            bpy.context.object.pose.bones["pinky_03_r"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_03_r"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['hand_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["hand_l"].constraints['Copy Location'].target = bpy.data.objects['00Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["hand_l"].constraints['Stretch To'].target = bpy.data.objects['09Hand Left']
            bpy.context.object.pose.bones["hand_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["hand_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["thumb_01_l"].constraints['Copy Location'].target = bpy.data.objects['01Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["thumb_01_l"].constraints['Stretch To'].target = bpy.data.objects['02Hand Left']
            bpy.context.object.pose.bones["thumb_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["thumb_02_l"].constraints['Copy Location'].target = bpy.data.objects['02Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["thumb_02_l"].constraints['Stretch To'].target = bpy.data.objects['03Hand Left']
            bpy.context.object.pose.bones["thumb_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['thumb_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["thumb_03_l"].constraints['Copy Location'].target = bpy.data.objects['03Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["thumb_03_l"].constraints['Stretch To'].target = bpy.data.objects['04Hand Left']
            bpy.context.object.pose.bones["thumb_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["thumb_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["index_01_l"].constraints['Copy Location'].target = bpy.data.objects['05Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["index_01_l"].constraints['Stretch To'].target = bpy.data.objects['06Hand Left']
            bpy.context.object.pose.bones["index_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["index_02_l"].constraints['Copy Location'].target = bpy.data.objects['06Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["index_02_l"].constraints['Stretch To'].target = bpy.data.objects['07Hand Left']
            bpy.context.object.pose.bones["index_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['index_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["index_03_l"].constraints['Copy Location'].target = bpy.data.objects['07Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["index_03_l"].constraints['Stretch To'].target = bpy.data.objects['08Hand Left']
            bpy.context.object.pose.bones["index_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["index_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["middle_01_l"].constraints['Copy Location'].target = bpy.data.objects['09Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["middle_01_l"].constraints['Stretch To'].target = bpy.data.objects['10Hand Left']
            bpy.context.object.pose.bones["middle_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["middle_02_l"].constraints['Copy Location'].target = bpy.data.objects['10Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["middle_02_l"].constraints['Stretch To'].target = bpy.data.objects['11Hand Left']
            bpy.context.object.pose.bones["middle_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['middle_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["middle_03_l"].constraints['Copy Location'].target = bpy.data.objects['11Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["middle_03_l"].constraints['Stretch To'].target = bpy.data.objects['12Hand Left']
            bpy.context.object.pose.bones["middle_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["middle_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["ring_01_l"].constraints['Copy Location'].target = bpy.data.objects['13Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["ring_01_l"].constraints['Stretch To'].target = bpy.data.objects['14Hand Left']
            bpy.context.object.pose.bones["ring_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["ring_02_l"].constraints['Copy Location'].target = bpy.data.objects['14Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["ring_02_l"].constraints['Stretch To'].target = bpy.data.objects['15Hand Left']
            bpy.context.object.pose.bones["ring_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['ring_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["ring_03_l"].constraints['Copy Location'].target = bpy.data.objects['15Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["ring_03_l"].constraints['Stretch To'].target = bpy.data.objects['16Hand Left']
            bpy.context.object.pose.bones["ring_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["ring_03_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_01_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["pinky_01_l"].constraints['Copy Location'].target = bpy.data.objects['17Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["pinky_01_l"].constraints['Stretch To'].target = bpy.data.objects['18Hand Left']
            bpy.context.object.pose.bones["pinky_01_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_01_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_02_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["pinky_02_l"].constraints['Copy Location'].target = bpy.data.objects['18Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["pinky_02_l"].constraints['Stretch To'].target = bpy.data.objects['19Hand Left']
            bpy.context.object.pose.bones["pinky_02_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_02_l"].constraints['Stretch To'].rest_length = 0.1

            bpy.context.object.data.bones.active = PosePipe_BodyBones.pose.bones['pinky_03_l'].bone
            bpy.ops.pose.constraint_add(type='COPY_LOCATION')
            bpy.context.object.pose.bones["pinky_03_l"].constraints['Copy Location'].target = bpy.data.objects['19Hand Left']
            bpy.ops.pose.constraint_add(type="STRETCH_TO")
            bpy.context.object.pose.bones["pinky_03_l"].constraints['Stretch To'].target = bpy.data.objects['20Hand Left']
            bpy.context.object.pose.bones["pinky_03_l"].constraints['Stretch To'].volume = 'NO_VOLUME'
            bpy.context.object.pose.bones["pinky_03_l"].constraints['Stretch To'].rest_length = 0.1

        hide_trackers = ['Body','Hand Left','Hand Right','Face',
                        '17 left pinky', '18 right pinky', '19 left index', 
                        '20 right index', '21 left thumb', '22 right thumb']

        for tracker in hide_trackers:
            bpy.data.objects[tracker].hide_set(True)

        face_trackers = ['01 left eye (inner)', '02 left eye', '03 left eye (outer)',
                        '04 right eye (inner)', '05 right eye', '06 right eye (outer)',
                        '09 mouth (left)', '10 mouth (right)']

        if settings.face_tracking:
            for tracker in face_trackers:
                bpy.data.objects[tracker].hide_set(True)

        return {'FINISHED'}

class PosePipePanel(Panel):
    bl_label = "PosePipe - Camera MoCap"
    bl_category = "PosePipe"
    bl_idname = "VIEW3D_PT_Pose"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):

        settings = context.scene.settings

        layout = self.layout
        
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Camera Settings:", icon='VIEW_CAMERA')
        column.operator(RunOperator.bl_idname, text="Start Camera", icon='CAMERA_DATA')
        split = column.split(factor=0.6)
        split.prop(settings, 'camera_number', text='Camera: ')
        split.label(text="to Exit", icon='EVENT_ESC')
        column.operator(RunFileSelector.bl_idname, text="Load Video File", icon='FILE_MOVIE')


        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Capture Mode:", icon='MOD_ARMATURE')
        column.prop(settings, 'body_tracking', text='Body', icon='ARMATURE_DATA')
        column.prop(settings, 'hand_tracking', text='Hands', icon='VIEW_PAN')
        column.prop(settings, 'face_tracking', text='Face', icon='MONKEY')
        column.label(text='Capture Settings:', icon='PREFERENCES')
        column.prop(settings, 'smoothing', text='Jitter Smoothing', icon='MOD_SMOOTH')
        column.prop(settings, 'model_complexity', text='Model Complexity:')
        column.prop(settings, 'detection_confidence', text='Detect Confidence:')
        column.prop(settings, 'tracking_confidence', text='Track Confidence:')
        
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Edit Capture Data:", icon='MODIFIER_ON')
        column.operator(RetimeAnimation.bl_idname, text="Retime Animation", icon='MOD_TIME')


        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Armature:", icon='BONE_DATA')
        column.operator(SkeletonBuilder.bl_idname, text="Generate Bones", icon='ARMATURE_DATA')
        
        
        
_classes = [
    Settings,
    PosePipePanel,
    RunOperator,
    RunFileSelector,
    SkeletonBuilder,
    RetimeAnimation,
]


def register():
    for c in _classes: register_class(c)
    bpy.types.Scene.settings = bpy.props.PointerProperty(type=Settings)


def unregister():
    for c in _classes: unregister_class(c)
    del bpy.types.Scene.settings


if __name__ == "__main__":
    register()
