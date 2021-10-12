from math import atan2

import numpy as np

from Arc import Arc


class SimpleArc(Arc):
    def at(self, t: float):
        return self.v1*t + self.v2*(1-t)


class LoopyArc(Arc):
    def __init__(self, v1: np.ndarray, radius: float, displacement: np.ndarray):
        super().__init__(v1, v1)
        displacement = displacement.astype(float)

        self.internal_radius = radius
        self.displacement = displacement

        norm_d = np.linalg.norm(displacement)
        self.external_radius = (norm_d * norm_d - radius * radius) / (2 * radius)

        dx, dy = displacement
        aux_displacement = np.asarray((-dy, dx))
        aux_displacement *= self.external_radius / norm_d
        self.lat_displacement = aux_displacement

        #midpoint = np.asarray((radius*self.external_radius, norm_d*self.external_radius))/(radius + self.external_radius)
        midpoint = np.asarray((norm_d, radius))*(self.external_radius/(radius + self.external_radius))
        displaced_midpoint = midpoint-np.asarray((norm_d, 0))
        alpha = atan2(*displaced_midpoint[::-1])
        self.internal_angle = 2*alpha
        self.external_angle = alpha-np.pi/2
        self.external_arc_length = self.external_angle*self.external_radius
        self.internal_arc_length = self.internal_angle*self.internal_radius
        self.arc_length = 2*self.external_arc_length + self.internal_arc_length

        self.starting_angle_1 = atan2(*((-self.lat_displacement)[::-1]))
        self.starting_angle_2 = atan2(*((self.lat_displacement-self.displacement)[::-1]))
        self.starting_angle_3 = atan2(*((self.lat_displacement+self.displacement)[::-1]))
        self.angle = atan2(*displacement)

    def at(self, t: float):
        t *= self.arc_length
        if t <= self.external_arc_length:
            angle = self.starting_angle_1 + self.external_angle*t/self.external_arc_length
            return self.v1 + self.lat_displacement + self.external_radius*np.asarray((np.cos(angle), np.sin(angle)))
        elif t <= self.external_arc_length+self.internal_arc_length:
            angle = self.starting_angle_2 \
                    - self.internal_angle*(t-self.external_arc_length)/self.internal_arc_length
            return self.v1 + self.displacement + self.internal_radius*np.asarray((np.cos(angle), np.sin(angle)))
        else:
            angle = self.starting_angle_3\
                    + self.external_angle*(t-self.external_arc_length-self.internal_arc_length)/self.external_arc_length
            return self.v1 - self.lat_displacement + self.external_radius*np.asarray((np.cos(angle), np.sin(angle)))


class SegmentedArc(Arc):
    def __init__(self, stops):
        super().__init__(stops[0], stops[-1])
        self.stops = stops
        self.qt_segments = len(stops)-1
        self.segment_lengths = [np.linalg.norm(stops[i]-stops[i-1]) for i in range(1,len(stops))]
        self.cumulative_lengths = np.cumsum([0,*self.segment_lengths])
        self.arc_length = self.cumulative_lengths[-1]

    def at(self, t: float):
        arc_t = t*self.arc_length
        mn, mx = 0, self.qt_segments
        while mx-mn>1:
            md = (mn+mx)//2
            if self.cumulative_lengths[md]>arc_t:
                mx = md
            else:
                mn = md
        p1, p2 = self.stops[mn:mx+1]
        local_t = (arc_t-self.cumulative_lengths[mn])/self.segment_lengths[mn]
        return p2*local_t + p1*(1-local_t)


class CircleArc(Arc):
    def __init__(self, p1, p2, center, reversed=False):
        super().__init__(p1, p2)
        self.center = center
        self.p1_h = np.linalg.norm(self.center-p1)
        self.p2_h = np.linalg.norm(self.center-p2)
        self.p1_a = atan2(*((p1-self.center)[::-1]))
        self.p2_a = atan2(*((p2-self.center)[::-1]))

        d1 = self.p1_a-self.p2_a
        d2 = 2*np.pi-d1
        if (d2<d1) != reversed:
            if self.p1_a<self.p2_a:
                self.p1_a += 2*np.pi
            else:
                self.p2_a += 2 * np.pi

    def at(self, t: float):
        h = (1-t)*self.p1_h + t*self.p2_h
        a = (1-t)*self.p1_a + t*self.p2_a
        return self.center+h*np.asarray((np.cos(a), np.sin(a)))
