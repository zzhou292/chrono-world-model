import pychrono as chrono
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class AssetsImporter:
    """
    A helper class to import various assets (ball, box, cylinder, table, mug, etc.) into the Chrono system.
    This class simplifies the process of loading meshes, creating bodies, and adding visual and 
    collision shapes for objects in the simulation.

     Attributes:
        system (chrono.ChSystemNSC): The PyChrono system to which the objects are added.
       
    """
    def __init__(self, system):
        """
        Constructor for the AssetsImporter class.
        """
        self.system = system
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(project_root)
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    def load_mesh(self, file_path, scale=1.0):
        """
        Helper function to load a mesh from a file and scale it.
        """
        mesh = chrono.ChTriangleMeshConnected()
        mesh.LoadWavefrontMesh(file_path, False, True)
        if scale != 1.0:
            mesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(scale))
        return mesh

    def create_body(self, mesh, position=chrono.ChVector3d(0, 0, 0), rotation=chrono.Q_ROTATE_Y_TO_Z, fixed=True, collidable=False, rolling_friction=0.005):
        """
        Helper function to create a body with a visual shape and an optional collision shape.
        """
        body = chrono.ChBody()
        body.SetPos(position)
        body.SetRot(rotation)
        body.SetFixed(fixed)

        # Add visual shape
        visual_shape = chrono.ChVisualShapeTriangleMesh()
        visual_shape.SetMesh(mesh)
        visual_shape.SetMutable(False)
        body.AddVisualShape(visual_shape)

        if collidable:
            # check collision system and create appropriate contact material
        
            if self.system.GetContactMethod() == 0:
                print("Using NSC contact material")
                contact_material = chrono.ChContactMaterialNSC()
            else:
                print("Using SMC contact material")
                contact_material = chrono.ChContactMaterialSMC()

            contact_material.SetRollingFriction(rolling_friction)
            collision_shape = chrono.ChCollisionShapeTriangleMesh(contact_material, mesh, False, False, 0.01)
            body.AddCollisionShape(collision_shape)
            body.EnableCollision(True)

        self.system.Add(body)
        return body

    def table(self, position=chrono.ChVector3d(0, 0, 0.45 - 0.98), collidable=True):
        """Import a table asset into the system, formed by a mesh loaded from a file.
        
        Args:
            position (chrono.ChVector3d): The position of the table.
            collidable (bool): Flag to enable collision for the table."""
        
        table_mesh = self.load_mesh(self.project_root + '/data/test_objs/table.obj', scale=0.01)
        return self.create_body(mesh=table_mesh, position=position, collidable=collidable)
    
    def box(self, dimension = [1,1,1], position=chrono.ChVector3d(0, 0, 0), rot = chrono.Q_ROTATE_Y_TO_Z, collidable=True):
        """Import a box asset into the system, formed by a mesh loaded from a file.
        
        Args:
            dimension (list): The dimensions of the box [length, width, height].
            position (chrono.ChVector3d): The position of the box.
            rot (chrono.ChQuaternion): The rotation of the box.
            collidable (bool): Flag to enable collision for the box."""
        
        if self.system.GetContactMethod() == 0:
            box_material = chrono.ChContactMaterialNSC()
        else:
            box_material = chrono.ChContactMaterialSMC()
        box = chrono.ChBodyEasyBox(dimension[0],dimension[1],dimension[2], 1000, True, True, box_material)
        box.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/bluewhite.png"))
        box.SetName("box")
        box.SetPos(position)
        box.SetRot(rot)
        box.SetFixed(False)
        box.EnableCollision(True)
        self.system.Add(box)
        return box

    def mug(self, position=chrono.ChVector3d(0, 0, 0), collidable=True, rolling_friction=0.01):
        """Import a mug asset into the system, formed by a mesh loaded from a file. Note that the mug's collision shape is a cylinder.
        while the visualization is a mesh.
        
        Args:
            position (chrono.ChVector3d): The position of the mug.
            collidable (bool): Flag to enable collision for the mug."""
        # initialize mug visual shape
        mug_mesh = chrono.ChTriangleMeshConnected()
        mug_mesh.LoadWavefrontMesh(self.project_root + '/data/test_objs/newMug.obj', False, True)
        mug_mesh.Transform(chrono.ChVector3d(0, -0.05, 0), chrono.ChMatrix33d(0.05))
        mug_shape = chrono.ChVisualShapeTriangleMesh()
        mug_shape.SetMesh(mug_mesh)
        mug_shape.SetMutable(False)


        if self.system.GetContactMethod() == 0:
            contact_material = chrono.ChContactMaterialNSC()
        else:
            contact_material = chrono.ChContactMaterialSMC()

        contact_material.SetRollingFriction(rolling_friction)


        mug = chrono.ChBody()
        mug.AddVisualShape(mug_shape) 
        mug.SetName("mug")

        # initialize mug collision shape

        mug_ct_shape = chrono.ChCollisionShapeTriangleMesh(contact_material, mug_mesh, False, False)
        mug.AddCollisionShape(mug_ct_shape)
        mug.EnableCollision(True)

        mug.SetPos(position)
        mug.SetRot(chrono.Q_ROTATE_Y_TO_Z)


        self.system.Add(mug)

        return mug


    # You can add more asset-loading functions similarly


