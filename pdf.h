#ifndef PDF_H
#define PDF_H
#include "vec3.h"
class pdf {
    public:
        virtual ~pdf() {}

        virtual double value(const vec3& direction) const = 0;
        virtual vec3 generate() const = 0;
};

class cosine_pdf : public pdf {
    public:
        cosine_pdf(const vec3& w, int idxQ) { 
            uvw.build_from_w(w);
            int numu = NUM_U,numv = NUM_V;
            double delu = 1.0/numu, delv = 1.0/numv;
            int uid = idxQ/numv,vid = idxQ%numv;
            v_min = ((double) vid)/numv; 
            u_min = ((double) uid)/numu;
            v_max = v_min + delv;
            u_max = u_min + delu;
        }

        virtual double value(const vec3& direction) const override {
            auto cosine = dot(unit_vector(direction), uvw.w());
            return (cosine <= 0) ? 0 : cosine/pi;
        }

        virtual vec3 generate() const override {
            if (RL_ON == 1) {
                double u = random_double(u_min, u_max), v = random_double(v_min, v_max);
                double phi = 2 * pi * v, temp = sqrt(1 - u * u);
                vec3 rand_dir(temp * cos(phi), temp * sin(phi), u);
                return uvw.local(rand_dir);
            }
            else
                 return uvw.local(random_cosine_direction());
        }

    public:
        onb uvw;
        double v_min,v_max,u_min,u_max;
};

class hittable_pdf : public pdf {
    public:
        hittable_pdf(shared_ptr<hittable> p, const point3& origin) : ptr(p), o(origin) {}

        virtual double value(const vec3& direction) const override {
            return ptr->pdf_value(o, direction);
        }

        virtual vec3 generate() const override {
            return ptr->random(o);
        }

    public:
        point3 o;
        shared_ptr<hittable> ptr;
};
class mixture_pdf : public pdf {
    public:
        mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1) {
            p[0] = p0;
            p[1] = p1;
        }

        virtual double value(const vec3& direction) const override {
             return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
            //return 0.5 *p[0]->value(direction);
        }

        virtual vec3 generate() const override {
            if (random_double() < 0.5)
                return p[0]->generate();
            else
                return p[1]->generate();
        }

    public:
        shared_ptr<pdf> p[2];
};
#endif