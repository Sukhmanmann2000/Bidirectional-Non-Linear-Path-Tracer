#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"
#include "texture.h"
#include "onb.h"
#include "pdf.h"

struct hit_record;
double schlick(double cosine, double ref_idx) {
    auto r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}
struct scatter_record {
    ray specular_ray;
    bool is_specular;
    color attenuation;
    shared_ptr<pdf> pdf_ptr;
};
class material {
    public:

       virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& albedo, ray& scattered, double& pdf, int idxQ
        ) const {
            return false;
        }

        virtual double scattering_pdf(
            const ray& r_in, const hit_record& rec, const ray& scattered
        ) const {
            return 0;
        }

        virtual color emitted(
            const ray& r_in, const hit_record& rec, double u, double v, const point3& p
        ) const {
            return color(0,0,0);
        }

        virtual int MatType() const { //To descibe material type to GPU
            return 0;
        }

        virtual color MatColor() const { //can be changed to std::vector materialProperties in the Future
            return color(0,0,0);
        }
};
class lambertian : public material {
    public:
        lambertian(const color& a) : albedo(make_shared<solid_color>(a)) {matColor = a;}
        lambertian(shared_ptr<texture> a) : albedo(a) {}

        virtual bool scatter(
		    const ray& r_in, const hit_record& rec, color& alb, ray& scattered, double& pdf, int idxQ
		) const override {
		    onb uvw;
		    uvw.build_from_w(rec.normal);
            vec3 direction;
            if (RL_ON == 1) {
                int numu = NUM_U, numv = NUM_V;
                double delu = 1.0 / numu, delv = 1.0 / numv;
                int uid = idxQ / numv, vid = idxQ % numv;
                double v_min = ((double)vid) / numv, u_min = ((double)uid) / numu;
                double v_max = v_min + delv, u_max = u_min + delu;
                double u = random_double(u_min, u_max), v = random_double(v_min, v_max);
                double phi = 2 * pi * v, temp = sqrt(1 - u * u);
                vec3 rand_dir(temp * cos(phi), temp * sin(phi), u);
                auto direction = uvw.local(rand_dir);
            }
            else
    		     auto direction = uvw.local(random_cosine_direction());
            scattered = ray(rec.p, unit_vector(direction), r_in.time());
		    alb = albedo->value(rec.u, rec.v, rec.p);
		    pdf = dot(uvw.w(), scattered.direction()) / pi;
		    return true;
		}
        double scattering_pdf(
            const ray& r_in, const hit_record& rec, const ray& scattered
        ) const {
            auto cosine = dot(rec.normal, unit_vector(scattered.direction()));
            return cosine < 0 ? 0 : cosine/pi;
        }
        virtual int MatType() const override{
            return 0;
        }
        virtual color MatColor() const override{
            return matColor;
        }

    public:
        shared_ptr<texture> albedo;
        color matColor;
};
 class metal : public material {
     public:
         metal(const color& a, double f) : albedo(make_shared<solid_color>(a)), fuzz(f < 1 ? f : 1) { matColor = a; }
         virtual bool scatter(
             const ray& r_in, const hit_record& rec, color& alb, ray& scattered, double& pdf, int idxQ
         ) const override {
             vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
             vec3 rand_dir;
             if (RL_ON == 1) {
                 int numu = NUM_U, numv = NUM_V;
                 double delu = 1.0 / numu, delv = 1.0 / numv;
                 int uid = idxQ / numv, vid = idxQ % numv;
                 double v_min = ((double)vid) / numv, u_min = ((double)uid) / numu;
                 double v_max = v_min + delv, u_max = u_min + delu;
                 double u = random_double(u_min, u_max), v = random_double(v_min, v_max);
                 double phi = 2 * pi * v, temp = sqrt(1 - u * u);
                 rand_dir = vec3(temp * cos(phi), temp * sin(phi), u);
             }
             else
                 vec3 rand_dir = random_in_unit_sphere();
             scattered = ray(rec.p, reflected+fuzz*rand_dir);
             alb = albedo->value(rec.u, rec.v, rec.p);
             pdf = 1;
             return true;
         }
         virtual int MatType() const override {
             return 2;
         }
         virtual color MatColor() const override {
             return matColor;
         }
     public:
         shared_ptr<texture> albedo;
         color matColor;
         double fuzz;
 };
 class dielectric : public material {
     public:
         dielectric(color matC,double ri) : matColor(matC),ref_idx(ri) {}

         virtual bool scatter(
             const ray& r_in, const hit_record& rec, color& alb, ray& scattered, double& pdf, int idxQ
         ) const override {
             alb = matColor;
             double etai_over_etat = rec.front_face ? (1.0 / ref_idx) : ref_idx;

             vec3 unit_direction = unit_vector(r_in.direction());
             double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
             double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
             if (etai_over_etat * sin_theta > 1.0 ) {
                 vec3 reflected = reflect(unit_direction, rec.normal);
                 scattered = ray(rec.p, reflected);
                 return true;
             }
             double reflect_prob = schlick(cos_theta, etai_over_etat);
             if (random_double() < reflect_prob)
             {
                 vec3 reflected = reflect(unit_direction, rec.normal);
                 scattered = ray(rec.p, reflected);
                 return true;
             }
             vec3 refracted = refract(unit_direction, rec.normal, etai_over_etat);
             scattered = ray(rec.p, refracted);
             pdf = 1;
             return true;
         }
         virtual int MatType() const override {
             return 3;
         }
         virtual color MatColor() const override {
             return matColor;
         }

     public:
         double ref_idx;
         color matColor;
 };
class diffuse_light : public material  {
    public:
        diffuse_light(shared_ptr<texture> a) : emit(a) {}
        diffuse_light(color c) : emit(make_shared<solid_color>(c)) {matColor = c;}

        virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& albedo, ray& scattered, double& pdf, int idxQ
        ) const override {
            return false;
        }

        virtual color emitted(const ray& r_in, const hit_record& rec, double u, double v,
    		const point3& p) const override {
            if (!rec.front_face)
		        return emit->value(u, v, p);
		    else
		        return color(0,0,0);
        }
        virtual int MatType() const override{
            return 1;
        }
        virtual color MatColor() const override{
            return matColor;
        }

    public:
        shared_ptr<texture> emit;
        color matColor;
};
// class isotropic : public material {
//     public:
//         isotropic(color c) : albedo(make_shared<solid_color>(c)) {}
//         isotropic(shared_ptr<texture> a) : albedo(a) {}

//         virtual bool scatter(
//             const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, double& pdf
//         ) const override {
//             scattered = ray(rec.p, random_in_unit_sphere(), r_in.time());
//             attenuation = albedo->value(rec.u, rec.v, rec.p);
//             return true;
//         }

//     public:
//         shared_ptr<texture> albedo;
// };
#endif