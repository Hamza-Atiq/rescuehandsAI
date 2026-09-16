# SO-101 robot source

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/8161bba264d7fa7c99ca301e91e7fb44737676ad/robotstudio_so101

Revision: 8161bba264d7fa7c99ca301e91e7fb44737676ad
License: Apache-2.0. Preserve robotstudio_so101/LICENSE with the downloaded assets.
The meshes and robot description are upstream work, not created by this project.
The runtime attaches two namespaced copies (left_arm/, right_arm/) and builds the
dinner-table scene around them: a table with a place mat, a static plate, a cup
(cylinder) and a fork and spoon made of simple boxes, all created by this project.
Item shapes are primitive; their sizes, mass, friction and positions vary per seed.
