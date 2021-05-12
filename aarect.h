#ifndef AARECT_H
#define AARECT_H

class xy_rect : public hittable {
    public:
        xy_rect() {}

        xy_rect(double _x0, double _x1, double _y0, double _y1, double _k, 
            shared_ptr<material> mat)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {
                qtable = new double*[NUM_POINTS];
                visits = new double*[NUM_POINTS];
                hammersley_points = new point3[NUM_POINTS];
                qmax = new double[NUM_POINTS];
                point3 points[NUM_POINTS];
                getHammerselyPoints(points,NUM_POINTS);
                for (int i=0;i<NUM_POINTS;i++){
                    hammersley_points[i].e[0]=x0+(x1-x0)*points[i].e[0];
                    hammersley_points[i].e[1]=y0+(y1-y0)*points[i].e[1];
                    hammersley_points[i].e[2]=k;
                    qtable[i] = new double[NUM_ACTIONS];
                    visits[i] = new double[NUM_ACTIONS];
                    double val = NUM_ACTIONS;
                    for (int j=0;j<NUM_ACTIONS;j++){
                        qtable[i][j]=1.0/val;
                        visits[i][j]=1;
                    }
                    qmax[i] = qtable[i][0];
                }
            };

        virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const override;

        virtual bool bounding_box(double t0, double t1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Z
            // dimension a small amount.
            output_box = aabb(point3(x0,y0, k-0.0001), point3(x1, y1, k+0.0001));
            return true;
        }
        virtual int clType() const override {
            return 1;
        }
        virtual std::vector<double> getInfo() const override {
            color matColor = mp->MatColor();
            std::vector<double> tmp{x0,x1,y0,y1,k,0.0,matColor.x(),matColor.y(),matColor.z(),cl_double(mp->MatType())};
            return tmp;
        }
        ~xy_rect(){
            for (int i=0;i<NUM_POINTS;i++){
                delete qtable[i];
                delete visits[i];
            }
            delete qtable;
            delete visits;
            delete hammersley_points;
            delete qmax;
        }

    public:
        shared_ptr<material> mp;
        double x0, x1, y0, y1, k;
        point3* hammersley_points;
        double **qtable,**visits;
        double* qmax;
};
class xz_rect : public hittable {
    public:
        xz_rect() {}

        xz_rect(double _x0, double _x1, double _z0, double _z1, double _k,
            shared_ptr<material> mat)
            : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {
                qtable = new double*[NUM_POINTS];
                visits = new double*[NUM_POINTS];
                hammersley_points = new point3[NUM_POINTS];
                qmax = new double[NUM_POINTS];
                point3 points[NUM_POINTS];
                getHammerselyPoints(points,NUM_POINTS);
                for (int i=0;i<NUM_POINTS;i++){
                    hammersley_points[i].e[0]=x0+(x1-x0)*points[i].e[0];
                    hammersley_points[i].e[2]=z0+(z1-z0)*points[i].e[1];
                    hammersley_points[i].e[1]=k;
                    qtable[i] = new double[NUM_ACTIONS];
                    visits[i] = new double[NUM_ACTIONS];
                    double val = NUM_ACTIONS;
                    for (int j=0;j<NUM_ACTIONS;j++){
                        qtable[i][j]=1.0/val;
                        visits[i][j]=1;
                    }
                    qmax[i] = qtable[i][0];
                }
            };

        virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const override;

        virtual bool bounding_box(double t0, double t1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Y
            // dimension a small amount.
            output_box = aabb(point3(x0,k-0.0001,z0), point3(x1, k+0.0001, z1));
            return true;
        }
        virtual double pdf_value(const point3& origin, const vec3& v) const override {
            hit_record rec;
            if (!this->hit(ray(origin, v), 0.001, infinity, rec))
                return 0;

            auto area = (x1-x0)*(z1-z0);
            auto distance_squared = rec.t * rec.t * v.length_squared();
            auto cosine = fabs(dot(v, rec.normal) / v.length());

            return distance_squared / (cosine * area);
        }

        virtual vec3 random(const point3& origin) const override {
            auto random_point = point3(random_double(x0,x1), k, random_double(z0,z1));
            return random_point - origin;
        }
        virtual int clType() const override {
            return 2;
        }
        virtual std::vector<double> getInfo() const override {
            color matColor = mp->MatColor();
            std::vector<double> tmp{x0,x1,z0,z1,k,0.0,matColor.x(),matColor.y(),matColor.z(),cl_double(mp->MatType())};
            return tmp;
        }
        ~xz_rect(){
            for (int i=0;i<NUM_POINTS;i++){
                delete qtable[i];
                delete visits[i];
            }
            delete qtable;
            delete visits;
            delete hammersley_points;
            delete qmax;
        }

    public:
        shared_ptr<material> mp;
        double x0, x1, z0, z1, k;
        point3* hammersley_points;
        double **qtable,**visits;
        double* qmax;
};

class yz_rect : public hittable {
    public:
        yz_rect() {}

        yz_rect(double _y0, double _y1, double _z0, double _z1, double _k,
            shared_ptr<material> mat)
            : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {
                qtable = new double*[NUM_POINTS];
                visits = new double*[NUM_POINTS];
                hammersley_points = new point3[NUM_POINTS];
                qmax = new double[NUM_POINTS];
                point3 points[NUM_POINTS];
                getHammerselyPoints(points,NUM_POINTS);
                for (int i=0;i<NUM_POINTS;i++){
                    hammersley_points[i].e[1]=y0+(y1-y0)*points[i].e[0];
                    hammersley_points[i].e[2]=z0+(z1-z0)*points[i].e[1];
                    hammersley_points[i].e[0]=k;
                    qtable[i] = new double[NUM_ACTIONS];
                    visits[i] = new double[NUM_ACTIONS];
                    double val = NUM_ACTIONS;
                    for (int j=0;j<NUM_ACTIONS;j++){
                        qtable[i][j]=1.0/val;
                        visits[i][j]=1;
                    }
                    qmax[i] = qtable[i][0];
                }
            };

        virtual bool hit(const ray& r, double t0, double t1, hit_record& rec) const override;

        virtual bool bounding_box(double t0, double t1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the X
            // dimension a small amount.
            output_box = aabb(point3(k-0.0001, y0, z0), point3(k+0.0001, y1, z1));
            return true;
        }
        virtual int clType() const override {
            return 3;
        }
        virtual std::vector<double> getInfo() const override {
            color matColor = mp->MatColor();
            std::vector<double> tmp{y0,y1,z0,z1,k,0.0,matColor.x(),matColor.y(),matColor.z(),cl_double(mp->MatType())};
            return tmp;
        }
        ~yz_rect(){
            for (int i=0;i<NUM_POINTS;i++){
                delete qtable[i];
                delete visits[i];
            }
            delete qtable;
            delete visits;
            delete hammersley_points;
            delete qmax;
        }

    public:
        shared_ptr<material> mp;
        double y0, y1, z0, z1, k;
        point3* hammersley_points;
        double **qtable,**visits;
        double* qmax;
};
bool xy_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const {
    auto t = (k-r.origin().z()) / r.direction().z();
    if (t < t0 || t > t1)
        return false;
    auto x = r.origin().x() + t*r.direction().x();
    auto y = r.origin().y() + t*r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;
    auto outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.qtable = qtable;
    rec.visits = visits;
    rec.qmax = qmax;
    rec.hammersely_points = hammersley_points;
    rec.p = r.at(t);
    return true;
}
bool xz_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const {
    auto t = (k-r.origin().y()) / r.direction().y();
    if (t < t0 || t > t1)
        return false;
    auto x = r.origin().x() + t*r.direction().x();
    auto z = r.origin().z() + t*r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.u = (x-x0)/(x1-x0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;
    auto outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.qtable = qtable;
    rec.visits = visits;
    rec.qmax = qmax;
    rec.hammersely_points = hammersley_points;
    rec.p = r.at(t);
    return true;
}

bool yz_rect::hit(const ray& r, double t0, double t1, hit_record& rec) const {
    auto t = (k-r.origin().x()) / r.direction().x();
    if (t < t0 || t > t1)
        return false;
    auto y = r.origin().y() + t*r.direction().y();
    auto z = r.origin().z() + t*r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.u = (y-y0)/(y1-y0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;
    auto outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.qtable = qtable;
    rec.visits = visits;
    rec.qmax = qmax;
    rec.hammersely_points = hammersley_points;
    rec.p = r.at(t);
    return true;
}
#endif