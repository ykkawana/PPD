import subprocess
import trimesh
import tempfile
import os


def gen_solidify_mesh_part(mesh, outfile=None):
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    assert isinstance(mesh, trimesh.Trimesh)

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = 'temp'
        infile = os.path.join(tempdir, 'infile.obj')
        mesh_str = trimesh.exchange.export.export_obj(mesh,
                                                      include_texture=False)
        with open(infile, 'w') as f:
            f.write(mesh_str)
        if outfile is None:
            outfile = os.path.join(tempdir, 'outfile.obj')
        command_args = [
            'symbolics/blender-2.79-linux-glibc219-x86_64/blender',
            '--background',
            '--python utils/blender/generate_solidify_part_mesh.py {infile} {outfile}'
            .format(infile=infile, outfile=outfile)
        ]
        command = ' '.join(command_args)
        print(command)
        subprocess.run(command, shell=True)
        rmesh = trimesh.load(outfile)
    return rmesh
