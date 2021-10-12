from math import atan2

import networkx as nx
import numpy as np
import cairo

from arcs import SimpleArc, SegmentedArc, LoopyArc, CircleArc


def getbounds(embedded_graph):
    mxx, mnx = -float("inf"), float("inf")
    mxy, mny = -float("inf"), float("inf")
    for node_index in embedded_graph:
        x, y = embedded_graph.nodes[node_index]["pos"]
        mxx = max(mxx, x)
        mnx = min(mnx, x)
        mxy = max(mxy, y)
        mny = min(mny, y)
    return {"mnx": mnx, "mxx": mxx, "mny": mny, "mxy": mxy}


def surface_to_ndarray(surface):
    buf = surface.get_data()
    array = np.ndarray (shape=(surface.get_height(),surface.get_width(),4), dtype=np.uint8, buffer=buf)
    return np.mean(array[:,:,:-1],axis=-1)


def draw(embedded_graph, resolution, bounds=None):
    if bounds is None:
        bounds = getbounds(embedded_graph)
    mnx, mxx, mny, mxy = map(bounds.__getitem__, ["mnx", "mxx", "mny", "mxy"])
    spanx, spany = mxx-mnx, mxy-mny
    margin = max(spanx,spany)*0.2
    mnx -= margin
    mxx += margin
    mny -= margin
    mxy += margin
    spanx += 2*margin
    spany += 2*margin

    im_width = int(spanx*resolution)
    im_height = int(spany*resolution)

    #surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, im_width, im_height)
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, im_width, im_height)
    ctx = cairo.Context(surface)
    ctx.scale(resolution, resolution)
    ctx.translate(-mnx, -mny)

    ctx.set_source_rgba(0, 0, 0, 1)
    ctx.rectangle(-spanx,-spany,2*spanx,2*spany)
    ctx.fill()

    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(0.005)
    for node_index in embedded_graph:
        pos = embedded_graph.nodes[node_index]["pos"]
        ctx.move_to(*pos)
        ctx.arc(*pos, 0.05, 0, 2*np.pi)
        ctx.fill()
    ctx.stroke()

    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(0.03)
    for v1,v2 in embedded_graph.edges:
        arc = embedded_graph.edges[v1, v2]["arc"]
        ctx.move_to(*arc.at(0))
        for t in np.linspace(0,1,100):
            ctx.line_to(*arc.at(t))
        ctx.stroke()

    return surface_to_ndarray(surface)


class AnimatedTransform:
    TOL = 1e-6

    def __call__(self, graph_1, graph_2, qt_steps, resolution, bounds=None):
        TOL = self.TOL
        sliding_function = lambda *p: self.sliding_function(*p)
        arc_sliding_function = lambda *p: self.arc_sliding_function(*p)

        static_node_idx = []
        moving_node_idx = []
        for node_index in graph_1:
            if np.linalg.norm(graph_1.nodes[node_index]["pos"]-graph_2.nodes[node_index]["pos"])>TOL:
                moving_node_idx.append(node_index)
            else:
                static_node_idx.append(node_index)

        static_arcs = []
        moving_arc_pairs = []
        for v1,v2 in graph_1.edges:
            arc_1 = graph_1.edges[v1, v2]["arc"]
            arc_2 = graph_2.edges[v1, v2]["arc"]
            #crappy, but it'll work for this project specifically
            p11, p12, p13 = map(arc_1.at, [0, 0.5, 1])
            p21, p22, p23 = map(arc_2.at, [0, 0.5, 1])
            max_d = max([np.linalg.norm(p11-p21),np.linalg.norm(p12-p22),np.linalg.norm(p13-p23)])
            if max_d>TOL:
                moving_arc_pairs.append((arc_1,arc_2))
            else:
                static_arcs.append(arc_1)

        if bounds is None:
            bounds1 = getbounds(graph_1)
            mnx1, mxx1, mny1, mxy1 = map(bounds1.__getitem__, ["mnx", "mxx", "mny", "mxy"])
            bounds2 = getbounds(graph_2)
            mnx2, mxx2, mny2, mxy2 = map(bounds2.__getitem__, ["mnx", "mxx", "mny", "mxy"])
            mnx = min(mnx1,mnx2)
            mxx = max(mxx1,mxx2)
            mny = min(mny1,mny2)
            mxy = max(mxy1,mxy2)
        else:
            mnx, mxx, mny, mxy = map(bounds.__getitem__, ["mnx", "mxx", "mny", "mxy"])
        spanx, spany = mxx-mnx, mxy-mny
        margin = max(spanx,spany)*0.2
        mnx -= margin
        mxx += margin
        mny -= margin
        mxy += margin
        spanx += 2*margin
        spany += 2*margin

        im_width = int(spanx*resolution)
        im_height = int(spany*resolution)

        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, im_width, im_height)
        ctx = cairo.Context(surface)
        ctx.scale(resolution, resolution)
        ctx.translate(-mnx, -mny)

        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.rectangle(-spanx,-spany,2*spanx,2*spany)
        ctx.fill()

        ctx.set_source_rgb(1, 1, 1)
        ctx.set_line_width(0.005)
        for node_index in static_node_idx:
            pos = graph_1.nodes[node_index]["pos"]
            ctx.move_to(*pos)
            ctx.arc(*pos, 0.05, 0, 2*np.pi)
            ctx.fill()
        ctx.stroke()

        ctx.set_source_rgb(1, 1, 1)
        ctx.set_line_width(0.03)
        for arc in static_arcs:
            ctx.move_to(*arc.at(0))
            for t in np.linspace(0,1,100):
                ctx.line_to(*arc.at(t))
            ctx.stroke()

        static_background = surface_to_ndarray(surface)
        frames= []

        for t in np.linspace(0,1,qt_steps):
            surface = cairo.ImageSurface(cairo.FORMAT_RGB24, im_width, im_height)
            ctx = cairo.Context(surface)
            ctx.scale(resolution, resolution)
            ctx.translate(-mnx, -mny)

            ctx.set_source_rgba(0, 0, 0, 1)
            ctx.rectangle(-spanx,-spany,2*spanx,2*spany)
            ctx.fill()

            ctx.set_source_rgb(1, 1, 1)
            ctx.set_line_width(0.005)
            for node_index in moving_node_idx:
                pos = sliding_function(graph_1.nodes[node_index]["pos"], graph_2.nodes[node_index]["pos"], t)
                ctx.move_to(*pos)
                ctx.arc(*pos, 0.05, 0, 2*np.pi)
                ctx.fill()
            ctx.stroke()

            ctx.set_source_rgb(1, 1, 1)
            ctx.set_line_width(0.03)
            for arc_1, arc_2 in moving_arc_pairs:
                pos = arc_sliding_function(arc_1, arc_2, t, 0)
                ctx.move_to(*pos)
                for tp in np.linspace(0,1,100):
                    pos = arc_sliding_function(arc_1, arc_2, t, tp)
                    ctx.line_to(*pos)
                ctx.stroke()

            new_stuff = surface_to_ndarray(surface)
            frames.append(np.maximum(static_background,new_stuff))
        return frames

    def sliding_function(self, p1, p2, t):
        raise NotImplementedError()

    def arc_sliding_function(self, arc1, arc2, transform_t, arc_t):
        return self.sliding_function(arc1.at(arc_t), arc2.at(arc_t), transform_t)


class SimpleAnimatedTransform(AnimatedTransform):
    def sliding_function(self, p1, p2, t):
        return (1-t)*p1 + t*p2


class SmoothAnimatedTransform(AnimatedTransform):
    def sliding_function(self, p1, p2, t):
        t = 1-np.cos(np.pi*t/2)
        return (1-t)*p1 + t*p2


class ExtraSmoothAnimatedTransform(AnimatedTransform):
    def sliding_function(self, p1, p2, t):
        t = 1-np.cos(np.pi*t/2)
        t = 1-np.cos(np.pi*t/2)
        return (1-t)*p1 + t*p2


class ArcSmoothAnimatedTransform(AnimatedTransform):
    def sliding_function(self, p1, p2, t):
        t = 1-np.cos(np.pi*t)/2
        return (1-t)*p1 + t*p2

    def arc_sliding_function(self, arc1, arc2, transform_t, arc_t):
        def f(x):
            if x<0: return 1
            if x>1: return 0
            return 0.5+np.cos(x*np.pi)/2
        t = f(arc_t-2*transform_t+1)
        return (1-t)*arc1.at(arc_t) + t*arc2.at(arc_t)


class CenteredAnimatedTransform(AnimatedTransform):
    def __init__(self, center):
        self.center = center

    def sliding_function(self, p1, p2, t):
        p1_h = np.linalg.norm(self.center-p1)
        p2_h = np.linalg.norm(self.center-p2)
        p1_a = atan2(*((p1-self.center)[::-1]))
        p2_a = atan2(*((p2-self.center)[::-1]))
        h = (1-t)*p1_h + t*p2_h
        a = (1-t)*p1_a + t*p2_a
        return h*(self.center+np.asarray((np.cos(a), np.sin(a))))



petersen = nx.petersen_graph()

normal_positions = [np.asarray((np.cos(alpha),np.sin(alpha))) for alpha in np.linspace(0, 2*np.pi, 5, endpoint=False)]

edges_outer = [SimpleArc(2*normal_positions[i], 2*normal_positions[i-1]) for i in range(5)]
edges_inner = [SimpleArc(normal_positions[i], normal_positions[i-2]) for i in range(5)]
edges_across = [SimpleArc(normal_positions[i], 2*normal_positions[i]) for i in range(5)]

for i in range(5):
    petersen.nodes[i]["pos"] = 2*normal_positions[i]
    petersen.nodes[i+5]["pos"] = normal_positions[i]
    petersen.edges[i+5,(i-2)%5+5]["arc"] = edges_inner[i]
    petersen.edges[i,(i-1)%5]["arc"] = edges_outer[i]
    petersen.edges[i,i+5]["arc"] = edges_across[i]


petersens = [petersen]
frames = []
for in_center in range(5, 10):
    in_right = (in_center + 3) % 5 + 5
    in_left = (in_center + 2) % 5 + 5
    out_center = in_center - 5
    out_right = in_right - 5
    out_left = in_left - 5

    petersen_1 = petersens[-1].copy()
    original_pos = petersen_1.nodes[in_center]["pos"]
    petersen_1.nodes[in_center]["pos"] = 0.45 * petersen_1.nodes[in_left]["pos"] \
                                         + 0.45 * petersen_1.nodes[in_right]["pos"] \
                                         + 0.1 * petersen_1.nodes[in_center]["pos"]
    petersen_1.edges[in_center, out_center]["arc"] = SimpleArc(petersen_1.nodes[in_center]["pos"],
                                                               petersen_1.nodes[out_center]["pos"])
    petersen_1.edges[in_center, in_right]["arc"] = SimpleArc(petersen_1.nodes[in_center]["pos"],
                                                             petersen_1.nodes[in_right]["pos"])
    petersen_1.edges[in_center, in_left]["arc"] = SimpleArc(petersen_1.nodes[in_left]["pos"],
                                                            petersen_1.nodes[in_center]["pos"])

    petersen_2 = petersen_1.copy()
    petersen_2.remove_edge(out_left, out_right)
    petersen_2.add_edge(out_right, out_right)
    petersen_2.edges[out_right, out_right]["arc"] = SegmentedArc([petersen_2.nodes[out_right]["pos"],
                                                                  petersen_2.nodes[out_left]["pos"],
                                                                  petersen_2.nodes[in_left]["pos"],
                                                                  petersen_2.nodes[in_center]["pos"],
                                                                  petersen_2.nodes[in_right]["pos"],
                                                                  petersen_2.nodes[out_right]["pos"],
                                                                  ])

    petersen_3 = petersen_2.copy()
    pos = petersen_3.nodes[out_right]["pos"]
    petersen_3.edges[out_right, out_right]["arc"] = LoopyArc(pos, 0.15, pos*0.2)

    petersen_4 = petersen_3.copy()
    petersen_4.nodes[in_center]["pos"] = original_pos
    petersen_4.edges[in_center, out_center]["arc"] = SimpleArc(original_pos, petersen_1.nodes[out_center]["pos"])
    petersen_4.edges[in_center, in_right]["arc"] = SimpleArc(original_pos, petersen_1.nodes[in_right]["pos"])
    petersen_4.edges[in_center, in_left]["arc"] = SimpleArc(petersen_1.nodes[in_left]["pos"], original_pos)

    last_petersen = petersens[-2] if len(petersens)>1 else petersens[-1]
    new_petersens = [petersen_1, petersen_2, petersen_3, petersen_4]
    petersens.extend(new_petersens)

    frames.extend(ExtraSmoothAnimatedTransform()(last_petersen, petersen_1, 30//(1 if in_center==5 else 2), 100))
    frames.extend(ArcSmoothAnimatedTransform()(petersen_2, petersen_3, 60//(1 if in_center==5 else 2), 100))

frames.extend(ExtraSmoothAnimatedTransform()(petersens[-2], petersens[-1], 30, 100))

last_petersen = petersens[-1]
petersen_unlooped = last_petersen.copy()
for i in range(5):
    petersen_unlooped.nodes[i]["pos"] = 2*normal_positions[(i*3)%5]
    petersen_unlooped.nodes[i+5]["pos"] = normal_positions[(i*3)%5]

    petersen_unlooped.edges[i+5,(i-2)%5+5]["arc"] = SimpleArc(normal_positions[(i*3)%5], normal_positions[(i*3-1)%5])
    petersen_unlooped.edges[i,i+5]["arc"] = edges_across[(i*3)%5]
    pos = petersen_unlooped.nodes[i]["pos"]
    petersen_unlooped.edges[i,i]["arc"] = LoopyArc(pos, 0.15, pos*0.2)

petersen_circle = petersen_unlooped.copy()
for i in range(5):
    petersen_circle.edges[i+5,(i-2)%5+5]["arc"] = CircleArc(normal_positions[(i*3-1)%5],
                                                            normal_positions[(i*3)%5],
                                                            np.asarray((0,0)))

petersen_fatbouquet = petersen_circle.copy()
normal_positions_18 = [np.asarray((np.cos(alpha), np.sin(alpha)))
                       for alpha in np.linspace(0, 2*np.pi, 18, endpoint=False)]
while len(normal_positions_18)>5:
    last_pos = normal_positions_18.pop(3)
original_center = normal_positions[0]
for i in range(5):
    ex_i = (3*i)%5
    petersen_fatbouquet.nodes[i+5]["pos"] = original_center
    petersen_fatbouquet.nodes[i]["pos"] = original_center + normal_positions_18[ex_i]
    petersen_fatbouquet.edges[i,i+5]["arc"] = SimpleArc(original_center, original_center + normal_positions_18[ex_i])
    petersen_fatbouquet.edges[i, i]["arc"] = LoopyArc(original_center + normal_positions_18[ex_i],
                                                      0.15, 0.2 * normal_positions_18[ex_i])
petersen_fatbouquet.edges[5, 7]["arc"] = CircleArc(original_center, original_center, np.asarray((0,0)))
petersen_fatbouquet.edges[7, 9]["arc"] = CircleArc(original_center, original_center, np.asarray((0,0)))
petersen_fatbouquet.edges[9, 6]["arc"] = CircleArc(original_center, original_center, np.asarray((0,0)), reversed=True)
petersen_fatbouquet.edges[6, 8]["arc"] = CircleArc(original_center, original_center, np.asarray((0,0)))
petersen_fatbouquet.edges[8, 5]["arc"] = CircleArc(original_center, original_center, np.asarray((0,0)))

petersen_fatbouquet2 = petersen_fatbouquet.copy()
for i in range(5):
    petersen_fatbouquet2.remove_node(i+5)
petersen_fatbouquet2.add_node(5, pos=original_center)
for i in range(5):
    ex_i = (3*i)%5
    petersen_fatbouquet2.add_edge(i, 5, arc=SimpleArc(original_center, original_center + normal_positions_18[ex_i]))
petersen_fatbouquet2.add_node(6, pos=original_center)
petersen_fatbouquet2.add_edge(5,6,arc=SimpleArc(original_center, original_center))
petersen_fatbouquet2.add_edge(6,6,arc=LoopyArc(original_center,0.99999,-original_center))

petersen_crappybouquet = petersen_fatbouquet2.copy()
petersen_crappybouquet.nodes[6]["pos"] = np.asarray((0,0))
petersen_crappybouquet.edges[5, 6]["arc"] = SimpleArc(original_center, np.asarray((0,0)))
petersen_crappybouquet.edges[6, 6]["arc"] = LoopyArc(np.asarray((0,0)),0.15,-0.2*original_center)

normal_positions_6 = [np.asarray((np.cos(alpha), np.sin(alpha)))
                       for alpha in np.linspace(0, 2*np.pi, 6, endpoint=False)]
normal_positions_6.pop(3)
for i in range(5):
    ex_i = (3*i)%5
    petersen_crappybouquet.nodes[i]["pos"] = original_center + normal_positions_6[ex_i]
    petersen_crappybouquet.edges[i, 5]["arc"] = SimpleArc(original_center, original_center + normal_positions_6[ex_i])
    petersen_crappybouquet.edges[i, i]["arc"] = LoopyArc(original_center + normal_positions_6[ex_i],
                                                      0.15, 0.2 * normal_positions_6[ex_i])


frames.extend(ExtraSmoothAnimatedTransform()(last_petersen, petersen_unlooped, 30, 100))
frames.extend(SmoothAnimatedTransform()(petersen_unlooped, petersen_circle, 30, 100))
frames.extend(CenteredAnimatedTransform(np.asarray((0,0)))(petersen_circle, petersen_fatbouquet, 50, 100))
frames.extend(SmoothAnimatedTransform()(petersen_fatbouquet2, petersen_crappybouquet, 50, 100, bounds=getbounds(petersen)))


import cv2
size = frames[0].shape

fps=24
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for frame in frames:
    out.write(frame.astype(np.uint8))
out.release()

for p in petersens:
    from matplotlib import pyplot as plt
    plt.imshow(draw(p, 100))
    plt.show()