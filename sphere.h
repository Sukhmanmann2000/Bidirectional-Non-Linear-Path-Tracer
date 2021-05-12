#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "material.h"
#include "vec3.h"
#include "aabb.h"
#include "onb.h"
class sphere : public hittable {
    public:
        sphere() {}
        sphere(point3 cen, double r, shared_ptr<material> m)
            : center(cen), radius(r), mp(m) {
            qtable = new double* [NUM_POINTS];
            visits = new double* [NUM_POINTS];
            hammersley_points = new point3[NUM_POINTS];
            qmax = new double[NUM_POINTS];
            point3 points[NUM_POINTS];
            getHammerselyPoints(points, NUM_POINTS);
            for (int i = 0; i < NUM_POINTS; i++) {
                double u = points[i].e[0], v = points[i].e[1];
                double temp = sqrt(1 - u * u);
                hammersley_points[i].e[0] = temp * cos(2 * pi * v);
                hammersley_points[i].e[1] = temp * sin(2 * pi * v);
                hammersley_points[i].e[2] = u;
                hammersley_points[i] = hammersley_points[i] + center;
                qtable[i] = new double[NUM_ACTIONS];
                visits[i] = new double[NUM_ACTIONS];
                double val = NUM_ACTIONS;
                for (int j = 0; j < NUM_ACTIONS; j++) {
                    qtable[i][j] = 1.0 / val;
                    visits[i][j] = 1;
                }
                qmax[i] = qtable[i][0];
            }
        };

        virtual bool hit(
            const ray& r, double tmin, double tmax, hit_record& rec) const override;
        virtual bool bounding_box(double t0, double t1, aabb& output_box) const override;
        virtual double pdf_value(const point3& o, const vec3& v) const;
        virtual vec3 random(const point3& o) const;
        virtual int clType() const override {
            return 5;
        }
        virtual std::vector<double> getInfo() const override {
            color matColor = mp->MatColor();
            std::vector<double> tmp{ center.x(),center.y(),center.z(),radius,0.0,0.0,matColor.x(),matColor.y(),matColor.z(),cl_double(mp->MatType()) };
            return tmp;
        }

    public:
        point3 center;
        double radius;
        shared_ptr<material> mp;
        point3* hammersley_points;
        double** qtable, ** visits;
        double* qmax;
};
void get_sphere_uv(const vec3& p, double& u, double& v) {
    auto phi = atan2(p.z(), p.x());
    auto theta = asin(p.y());
    u = 1-(phi + pi) / (2*pi);
    v = (theta + pi/2) / pi;
}
inline vec3 random_to_sphere(double radius, double distance_squared) {
    auto r1 = random_double();
    auto r2 = random_double();
    auto z = 1 + r2*(sqrt(1-radius*radius/distance_squared) - 1);

    auto phi = 2*pi*r1;
    auto x = cos(phi)*sqrt(1-z*z);
    auto y = sin(phi)*sqrt(1-z*z);

    return vec3(x, y, z);
}
bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = half_b*half_b - a*c;

    if (discriminant > 0) {
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
            rec.mat_ptr = mp;
            rec.qtable = qtable;
            rec.visits = visits;
            rec.qmax = qmax;
            rec.hammersely_points = hammersley_points;
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
            rec.mat_ptr = mp;
            rec.qtable = qtable;
            rec.visits = visits;
            rec.qmax = qmax;
            rec.hammersely_points = hammersley_points;
            return true;
        }
    }
    return false;
}
bool sphere::bounding_box(double t0, double t1, aabb& output_box) const {
    output_box = aabb(
        center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));
    return true;
}
double sphere::pdf_value(const point3& o, const vec3& v) const {
    hit_record rec;
    if (!this->hit(ray(o, v), 0.001, infinity, rec))
        return 0;

    auto cos_theta_max = sqrt(1 - radius*radius/(center-o).length_squared());
    auto solid_angle = 2*pi*(1-cos_theta_max);

    return  1 / solid_angle;
}

vec3 sphere::random(const point3& o) const {
    vec3 direction = center - o;
    auto distance_squared = direction.length_squared();
    onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(random_to_sphere(radius, distance_squared));
}
#endif