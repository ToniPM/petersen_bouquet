# petersen_bouquet
A small project to show that the petersen graph can be continuously deformed into a bouquet of 6 1-spheres, made for a practical of Discrete and Algorithmic Geometry, a course from MAMME-UPC.

To run, execute main.py, with the "graph animations" directory set as the sources root (alternatively, run main.py directly after moving it inside the "graph animations" dir).

The code isn't written to be particularly reusable, but some parts might be suitable for use somewhere else. If you intend to use any of the code here, note the comment at line 96 of main.py --- to check if two arcs are equal, the program simply checks if they're close at some points along the arc. This works for how the arcs are used in this project specifically, but might fail for other use cases.


