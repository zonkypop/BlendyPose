bl_info = {
    "name": "PosePipe",
    "author": "ZonkoSoft, SpectralVectors",
    "version": (0, 6),
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
                        bones[k].location.y = bns[k].z * 2
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.2-bns[k].y) + 2
                        bones[k].keyframe_insert(data_path="location", frame=n)


                if results.right_hand_landmarks:
                    bns = [b for b in results.right_hand_landmarks.landmark]
                    scale = 2
                    bones = sorted(hand_right.children, key=lambda b: b.name)
                    for k in range(21):
                        bones[k].location.y = bns[k].z * 2
                        bones[k].location.x = (0.5-bns[k].x)
                        bones[k].location.z = (0.2-bns[k].y) + 2
                        bones[k].keyframe_insert(data_path="location", frame=n)


            if settings.face_tracking:
                if results.face_landmarks:
                    bns = [b for b in results.face_landmarks.landmark]
                    scale = 2
                    bones = sorted(face.children, key=lambda b: b.name)
                    for k in range(468):
                        bones[k].location.y = bns[k].z * 2
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
    bpy.context.view_layer.objects.active = bpy.data.objects['Face']
    bpy.ops.object.constraint_add(type='COPY_LOCATION')
    bpy.data.objects['Face'].constraints['Copy Location'].target = bpy.data.objects['Pose']
    bpy.data.objects['Face'].constraints["Copy Location"].use_y = False
    bpy.ops.object.constraint_add(type='COPY_LOCATION')
    bpy.data.objects['Face'].constraints['Copy Location.001'].target = bpy.data.objects['00 nose']
    bpy.data.objects['Face'].constraints["Copy Location.001"].use_x = False
    bpy.data.objects['Face'].constraints["Copy Location.001"].use_z = False

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
        # Pelvis and Spine
        bpy.ops.object.armature_add(radius=0.3)
        pelvis = bpy.context.object
        bpy.context.object.name = 'pelvis'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['23 left hip']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location.001'].target = bpy.data.objects['24 right hip']
        bone.constraints['Copy Location.001'].influence = 0.5

        #bpy.ops.object.editmode_toggle()
        #bpy.ops.view3d.snap_cursor_to_selected()
        #bpy.ops.object.editmode_toggle()

        #bpy.ops.object.armature_add(radius=0.3)
        #spine01 = bpy.context.object
        #bpy.context.object.name = 'spine01'
        #bonename = bpy.context.object.name
        #bone = bpy.data.objects[bonename]

        #bpy.ops.object.editmode_toggle()
        #bpy.ops.view3d.snap_cursor_to_selected()
        #bpy.ops.object.editmode_toggle()

        #bpy.ops.object.armature_add(radius=0.3)
        #spine02 = bpy.context.object
        #bpy.context.object.name = 'spine02'
        #bonename = bpy.context.object.name
        #bone = bpy.data.objects[bonename]

        #bpy.ops.object.editmode_toggle()
        #bpy.ops.view3d.snap_cursor_to_selected()
        #bpy.ops.object.editmode_toggle()

        #bpy.ops.object.armature_add(radius=0.3)
        #spine03 = bpy.context.object
        #bpy.context.object.name = 'spine03'
        #bonename = bpy.context.object.name
        #bone = bpy.data.objects[bonename]

        bpy.ops.object.armature_add(radius=0.3)
        neck = bpy.context.object
        bpy.context.object.name = 'neck'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['11 left shoulder']
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location.001'].target = bpy.data.objects['12 right shoulder']
        bone.constraints['Copy Location.001'].influence = 0.5

        # Left Arm
        bpy.ops.object.armature_add()
        clavicle_l = bpy.context.object
        bpy.context.object.name = 'clavicle_l'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['11 left shoulder']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['12 right shoulder']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 2
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bone.pose.bones['Bone'].constraints['Stretch To'].keep_axis = 'PLANE_Z'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        upperarm_l = bpy.context.object
        bpy.context.object.name = 'upperarm_l'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['11 left shoulder']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['13 left elbow']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        lowerarm_l = bpy.context.object
        bpy.context.object.name = 'lowerarm_l'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['13 left elbow']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['15 left wrist']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()


        # Right Arm
        bpy.ops.object.armature_add()
        clavicle_r = bpy.context.object
        bpy.context.object.name = 'clavicle_r'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['12 right shoulder']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['11 left shoulder']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 2
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bone.pose.bones['Bone'].constraints['Stretch To'].keep_axis = 'PLANE_Z'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        upperarm_r = bpy.context.object
        bpy.context.object.name = 'upperarm_r'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['12 right shoulder']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['14 right elbow']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        lowerarm_r = bpy.context.object
        bpy.context.object.name = 'lowerarm_r'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['14 right elbow']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['16 right wrist']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        # Left Leg
        bpy.ops.object.armature_add()
        thigh_l = bpy.context.object
        bpy.context.object.name = 'thigh_l'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['23 left hip']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['25 left knee']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        calf_l = bpy.context.object
        bpy.context.object.name = 'calf_l'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['25 left knee']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['27 left ankle']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        foot_l = bpy.context.object
        bpy.context.object.name = 'foot_l'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['27 left ankle']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['31 left foot index']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        # Right Leg
        bpy.ops.object.armature_add()
        thigh_r = bpy.context.object
        bpy.context.object.name = 'thigh_r'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['24 right hip']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['26 right knee']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        calf_r = bpy.context.object
        bpy.context.object.name = 'calf_r'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['26 right knee']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['28 right ankle']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()

        bpy.ops.object.armature_add()
        foot_r = bpy.context.object
        bpy.context.object.name = 'foot_r'
        bpy.context.object.data.display_type = 'STICK'
        bonename = bpy.context.object.name
        bone = bpy.data.objects[bonename]
        bpy.ops.object.constraint_add(type='COPY_LOCATION')
        bone.constraints['Copy Location'].target = bpy.data.objects['28 right ankle']
        bpy.ops.object.posemode_toggle()
        bpy.ops.pose.constraint_add(type="STRETCH_TO")
        bone.pose.bones['Bone'].constraints['Stretch To'].target = bpy.data.objects['32 right foot index']
        bone.pose.bones['Bone'].constraints['Stretch To'].rest_length = 1
        bone.pose.bones['Bone'].constraints['Stretch To'].volume = 'NO_VOLUME'
        bpy.ops.object.posemode_toggle()


        # Parenting Bone Chains

        thigh_r.parent = pelvis
        thigh_r.parent_type = 'ARMATURE'

        thigh_l.parent = pelvis
        thigh_r.parent_type = 'ARMATURE'

        neck.parent = pelvis
        neck.parent_type = 'ARMATURE'

        clavicle_r.parent = neck
        clavicle_r.parent_type = 'ARMATURE'

        clavicle_l.parent = neck
        clavicle_l.parent_type = 'ARMATURE'

        upperarm_r.parent = clavicle_r
        upperarm_r.parent_type = 'ARMATURE'

        upperarm_l.parent = clavicle_l
        upperarm_l.parent_type = 'ARMATURE'

        lowerarm_r.parent = upperarm_r
        lowerarm_r.parent_type = 'ARMATURE'

        lowerarm_l.parent = upperarm_l
        lowerarm_l.parent_type = 'ARMATURE'

        calf_r.parent = thigh_r
        calf_r.parent_type = 'ARMATURE'

        foot_r.parent = calf_r
        foot_r.parent_type = 'ARMATURE'

        calf_l.parent = thigh_l
        calf_l.parent_type = 'ARMATURE'

        foot_l.parent = calf_l
        foot_l.parent_type = 'ARMATURE'

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

        #row = layout.row()

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

        #row = layout.row()        
        
        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Edit Capture Data:", icon='MODIFIER_ON')
        column.operator(RetimeAnimation.bl_idname, text="Retime Animation", icon='MOD_TIME')

        #row = layout.row()

        box = layout.box()
        column_flow = box.column_flow()
        column = column_flow.column(align=True)
        column.label(text="Generate Armature:", icon='BONE_DATA')
        column.operator(SkeletonBuilder.bl_idname, text="Body Bones", icon='ARMATURE_DATA')
        
        
        
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
