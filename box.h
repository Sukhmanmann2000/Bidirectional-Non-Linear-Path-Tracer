#ifndef BOX_H
#define BOX_H

#include "rtweekend.h"

#include "aarect.h"
#include "hittable_list.h"

class box : public hittable  { //Support for this remains to be added for RL, but can be done using rectangles
    public:
        box() {}
        box(const point3& p0, const point3& p1, shared_ptr<material> ptr);

        virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const override;

        virtual bool bounding_box(double t0, double t1, aabb& output_box) const override {
            output_box = aabb(box_min, box_max);
            return true;
        }
        virtual int clType() const override {
            return 4;
        }
        virtual std::vector<double> getInfo() const override {
            color matColor = mp->MatColor();
            std::vector<double> tmp{box_min.x(),box_min.y(),box_min.z(),box_max.x(),box_max.y(),box_max.z(),matColor.x(),matColor.y(),matColor.z(),cl_double(mp->MatType())};
            return tmp;
        }

    public:
        point3 box_min;
        point3 box_max;
        hittable_list sides;
        shared_ptr<material> mp;
};

box::box(const point3& p0, const point3& p1, shared_ptr<material> ptr) {
    mp = ptr;
    box_min = p0;
    box_max = p1;

    sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
    sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

    sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
    sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

    sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
    sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

bool box::hit(const ray& r, double t0, double t1, hit_record& rec) const {
    return sides.hit(r, t0, t1, rec);
}

#endif