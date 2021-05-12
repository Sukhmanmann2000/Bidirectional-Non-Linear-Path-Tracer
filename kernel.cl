#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//To render the random_scene, choose the background from the commented options,
//Then change the world and mats array length to 500.
__constant int MAX_DEPTH = 10;
__constant int NUM_POINTS = 100;
__constant int USE_HAMM = 0;
__constant int NUM_U = 8;
__constant int NUM_V = 16;
__constant int NUM_ACTIONS = 128;
__constant int RL_ON=0;  //0 for without RL and 1 for rendering with RL ON
__constant int BIDIR_ON=1;
__constant int NON_LINEAR_ON=0;
__constant double scale = 1.0;
__constant double FUZZ = 0.01;
__constant double3 background = (double3){0.0,0.0,0.0};
// __constant double3 background = (double3){0.05,0.05,0.05};
// __constant double3 background = (double3){0.68, 0.84, 0.9};

// Define C++ Classes as OpenCL structs
typedef struct _cl_tag_sObject {
	double x0,x1,y0,y1,z0,z1,k,radius;
	int m_type;
	int off;
	int numPoints;
} sObject;
typedef struct _cl_tag_Ray {
	double3 a;
	double3 b;
} Ray;
typedef struct _cl_tag_Camera {
	double3 lookFrom;
	double3 lookAt;
	double3 viewUp;
	double aperture;
	double Fov;
	double focus_dist;
} Camera;
typedef struct _cl_tag_Material {
	double3 m_vColor;
	bool flip_face;
	int m_MType;
} Material;

typedef struct _cl_tag_HitRecord {
	double3 p;
	double3 normal;
	Material m_curmat;
	double t,u,v;
	bool front_face;
	int objId;
} HitRecord;

typedef struct _cl_Axes {
	double3 s;
	double3 t;
	double3 n;
} Axes;

typedef struct _cl_PathVertex {
	HitRecord rec;
	double3 throughput;
	double vcm;
	double vc;
	double3 wo;
} PathVertex;
void set_face_normal(Ray r,double3 outward_normal,HitRecord *rec){
	rec->front_face = dot(r.b,outward_normal) < 0;
	if (rec->front_face)
		rec->normal = outward_normal;
	else
		rec->normal = -outward_normal;
}
// Define math functions
double3 UnitVector(double3 v) {
	return v / length(v);
}
double3 PointAtParameter(Ray r, double t) { 
	return r.a + t * r.b; 
}
double3 InvDir(const Ray r) {
	return 1 / r.b;
}
double SquaredLength(double3 m_dE) {
	return m_dE.x * m_dE.x + m_dE.y * m_dE.y + m_dE.z * m_dE.z;
}
double SquaredLength2(double2 m_dE) {
	return m_dE.x * m_dE.x + m_dE.y * m_dE.y;
}
uint MWC64X(uint2 *state)
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2){x,c};            // Pack the state back up
    return res;                       // Return the next result
}
double myrand(uint2* seed_ptr) //Random Double
{
	uint MAX_INT=0;
	MAX_INT--;
	uint rInt = MWC64X(seed_ptr);
    double ans = ((double)rInt)/MAX_INT;
    return ans;
 }
 double3 random_in_unit_disk(uint2* seed_ptr) {
    while (true) {
        double3 p = (double3){2*myrand(seed_ptr)-1,2*myrand(seed_ptr)-1,0};
        if (SquaredLength(p) >= 1) continue;
        return p;
    }
}
double2 get_sphere_uv(double3 p) {  //texture map of sphere
	double2 ans;
	double phi = atan2(p.z, p.x);
	double theta = asin(p.y);
	ans.x = 1-(phi + M_PI) / (2*M_PI);
	ans.y = (theta + M_PI/2) / M_PI;
	return ans;
}
double2 get_obj_uv(sObject obj,double3 p){  //texture map of all objects
	double2 uv;
	if (obj.m_type == 1){
		uv.x = (p.x-obj.x0)/(obj.x1-obj.x0);
		uv.y = (p.y-obj.y0)/(obj.y1-obj.y0);
	}
	else if (obj.m_type == 2){
		uv.x = (p.x-obj.x0)/(obj.x1-obj.x0);
		uv.y = (p.z-obj.z0)/(obj.z1-obj.z0);
	}
	else if (obj.m_type == 3){
		uv.x = (p.y-obj.y0)/(obj.y1-obj.y0);
		uv.y = (p.z-obj.z0)/(obj.z1-obj.z0);
	}
	else if (obj.m_type == 4){
		double3 center = (double3){obj.x0,obj.y0,obj.z0};
		uv = get_sphere_uv((p-center)/obj.radius);
	}
	return uv;
}
int getClosestIndexHamm(global const double2* points, sObject obj,double3 p){  //Linear search for closest point
	double2 uv = get_obj_uv(obj,p);
    int ans=-1;
    double mindis=0;
    for (int i=0;i<obj.numPoints;i++){
		double2 temp = points[i+obj.off] - uv;
        double dis = sqrt(SquaredLength2(temp));
        if (ans==-1){
            ans = i;
            mindis = dis;
        }
        else if (dis < mindis){
            mindis = dis;
            ans = i;
        }
    }
    return ans;
}
int getClosestIndexGrid(sObject obj,double3 p){  //Linear search for closest point
	double2 uv = get_obj_uv(obj,p);
    int xid = min((int)(uv.y*obj.numPoints),obj.numPoints-1);
	int yid = min((int)(uv.x*obj.numPoints),obj.numPoints-1);
	int ans = yid*obj.numPoints + xid;
    return ans;
}
double3 random_cosine_direction(uint2* seed_ptr){
	double r1 = myrand(seed_ptr);
	double r2 = myrand(seed_ptr);
	double z = r2;
	double phi = 2*M_PI*r1;
	double sinTheta = sqrt(max(0.0,1-r2*r2));
	double x = cos(phi)*sinTheta;
	double y = sin(phi)*sinTheta;
	return (double3){x,y,z};
}

// Define external functions
double myabs(double x){
	if (x<0)
		x=-x;
	return x;
}
double3 Reflect(const double3 v, const double3 n) {
	return (double3)(v - 2 * dot(v, n)*n);
}
double3 Refract(const double3 uv, const double3 n, double etai_over_etat) {
	double cos_theta = min(dot(-uv, n), 1.0);
    double3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    double3 r_out_parallel = -sqrt(myabs(1.0 - SquaredLength(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}
double schlick(double cosine, double ref_idx) {
	double d0 = (1 - ref_idx) / (1 + ref_idx);
	d0 = d0 * d0;
	return d0 + (1 - d0)*pow((1 - cosine), 5);
}
double3 get_rand_dir(int idx_Q,uint2* seed_ptr, bool center){  //Random direction according to idxQ if center is false
	int numu = NUM_U, numv = NUM_V;
	double delu = 1.0/numu, delv = 1.0/numv;
	int uid = idx_Q/numv, vid = idx_Q%numv;
	double v_min = vid*delv, u_min = uid*delu;
	double r1,r2;
	if (center){
		r1 = 0.5;
		r2 = 0.5;
	}
	else{
		r1 = myrand(seed_ptr);
		r2 = myrand(seed_ptr);
	}
	double u = u_min + delu*r1;
	double v = v_min + delv*r2;
	double phi = 2*M_PI*v, temp = sqrt(1-u*u);
	double3 rand_dir = (double3){temp*cos(phi),temp*sin(phi),u};
	return rand_dir;
}
Axes getAxes(double3 normal){
	Axes ans;
	ans.n = UnitVector(normal);
	if (myabs(ans.n.x) > myabs(ans.n.y)) {
        double invLen = 1.0 / sqrt(ans.n.x * ans.n.x + ans.n.z * ans.n.z);
        ans.t = (double3){ans.n.z * invLen, 0.0, -ans.n.x * invLen};
    } else {
        double invLen = 1.0 / sqrt(ans.n.y * ans.n.y + ans.n.z * ans.n.z);
        ans.t = (double3){0.0, ans.n.z * invLen, -ans.n.y * invLen};
    }
    ans.s = cross(ans.t, ans.n);
	return ans;
}
double3 get_direction(double3 normal,int idx_Q,uint2* seed_ptr, bool center){
	Axes coAxes;
	coAxes.n = UnitVector(normal);
	double3 a;
	if (myabs(coAxes.n.x)>0.9)
		a=(double3){0,1,0};
	else
		a=(double3){1,0,0};
	coAxes.t = UnitVector(cross(coAxes.n,a));
	coAxes.s = cross(coAxes.n,coAxes.t);
	double3 rand_dir;
	if (RL_ON==1)
		rand_dir = get_rand_dir(idx_Q,seed_ptr,center);
	else
		rand_dir = random_cosine_direction(seed_ptr);
	
	double3 direction = rand_dir.x*coAxes.s + rand_dir.y*coAxes.t + rand_dir.z*coAxes.n;
	return UnitVector(direction);
}
double3 getObjColor(HitRecord rec){
	// double3 offset = (double3){130*scale,168*scale,-30*scale};
	double3 offset = (double3){0,0,0};
	if (rec.objId==5){
		double xcoor = rec.p.x - offset.x, ycoor = rec.p.y - offset.y, hwidth = 3*scale, wallhw = 278*scale;
		int numDivs = 5;
		bool condition = false;
		for (int i=1;i<numDivs;i++){
			double currx = -wallhw + 2.0*i*wallhw/numDivs;
			if ((xcoor>currx-hwidth && xcoor<currx+hwidth) ||
				(ycoor>currx-hwidth && ycoor<currx+hwidth)){
					condition = true;
					break;
				}
		}
		if (condition)
			return (double3){0.1,0.1,0.1};
		// else{
		// 	int xweight = (wallhw+xcoor)*numDivs/(2*wallhw);
		// 	int yweight = (wallhw+ycoor)*numDivs/(2*wallhw);
		// 	double weight = 1.0*(xweight + yweight)/(2.0*numDivs);
		// 	double3 rcolor = (double3){.65, .05, .05}, lcolor = (double3){.12, .45, .15};
		// 	*attenuation = (1 - weight)*rcolor + weight*lcolor;
		// }
	}
	return rec.m_curmat.m_vColor;
}
bool scatter(Material *mat, const Ray r_in, const HitRecord *rec, double3 *attenuation, double *pdf_val, Ray *scattered, uint2* seed_ptr, int idx_Q) { //combined scatter function
	// Lambertian
	if (mat->m_MType == 0) {
		double3 direction = get_direction(rec->normal,idx_Q,seed_ptr,false);
		*scattered = (Ray){rec->p,direction};
		*attenuation = getObjColor(*rec);
		*pdf_val = dot(UnitVector(rec->normal),scattered->b)/M_PI;

		
		return true;
	}
	// Light
	else if (mat->m_MType == 1) {
		return false;
	}
	// Metal
	else if (mat->m_MType == 2) {
		double3 reflected = Reflect(UnitVector(r_in.b),rec->normal);

		double3 rand_dir;
		if (RL_ON==1)
			rand_dir = get_rand_dir(idx_Q,seed_ptr,false);
		else
			rand_dir = random_cosine_direction(seed_ptr);

		double fuzz = FUZZ;
		*scattered = (Ray) {rec->p, UnitVector(reflected+fuzz*rand_dir)};
		*attenuation = mat->m_vColor;
		*pdf_val = 1;
		return true;
	}
	//Dielectric
	else if (mat->m_MType == 3) {
		*attenuation = mat->m_vColor;
		double ref_idx = 1.5;
		double etai_over_etat = rec->front_face ? (1.0 / ref_idx) : ref_idx;
		double3 unit_direction = UnitVector(r_in.b);
		double cos_theta = min(dot(-unit_direction, rec->normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

		double3 rand_dir;
		if (RL_ON==1)
			rand_dir = get_rand_dir(idx_Q,seed_ptr,false);
		else
			rand_dir = random_cosine_direction(seed_ptr);
		double fuzz = 0.01;

		if (etai_over_etat * sin_theta > 1.0 ) {
			double3 reflected = Reflect(unit_direction, rec->normal);
			*scattered = (Ray) {rec->p, UnitVector(reflected+fuzz*rand_dir)};
			*pdf_val = 1;
			return true;
		}
		double reflect_prob = schlick(cos_theta, etai_over_etat);
		if (myrand(seed_ptr) < reflect_prob)
		{
			double3 reflected = Reflect(unit_direction, rec->normal);
			*scattered = (Ray) {rec->p, UnitVector(reflected+fuzz*rand_dir)};
			*pdf_val = 1;
			return true;
		}
		double3 refracted = Refract(unit_direction, rec->normal, etai_over_etat);
		*scattered = (Ray) {rec->p, refracted};
		*pdf_val = 1;
		return true;
	}
	printf("Hi1\n");
	return false;
}
double3 getPointAtuv(double u, double v, const sObject obj){
	double3 ans;
	if (obj.m_type == 1){
		ans.x = obj.x0 + (obj.x1 - obj.x0)*u;
		ans.y = obj.y0 + (obj.y1 - obj.y0)*v;
		ans.z = obj.k;
	}
	else if (obj.m_type == 2) {
		ans.x = obj.x0 + (obj.x1 - obj.x0)*u;
		ans.z = obj.z0 + (obj.z1 - obj.z0)*v;
		ans.y = obj.k;
	}
	else if (obj.m_type == 3) {
		ans.y = obj.y0 + (obj.y1 - obj.y0)*u;
		ans.z = obj.z0 + (obj.z1 - obj.z0)*v;
		ans.x = obj.k;
	}
	else if (obj.m_type == 5) {
		double phi = 2*M_PI*v, theta = M_PI*u;
		ans.x = obj.x0 + obj.radius*cos(phi)*sin(theta);
		ans.y = obj.y0 + obj.radius*sin(phi)*sin(theta);
		ans.z = obj.z0 + obj.radius*cos(theta);
	}
	return ans;
}
double3 getNormalAtuv(double u, double v, const sObject obj, const Material mat){
	double3 ans;
	double flipped = 1;
	if (mat.flip_face)
		flipped = -1;
	if (obj.m_type == 1){
		ans = (double3){0,0,flipped};
	}
	else if (obj.m_type == 2) {
		ans = (double3){0,flipped,0};
	}
	else if (obj.m_type == 3) {
		ans = (double3){flipped,0,0};
	}
	else if (obj.m_type == 5) {
		double phi = 2*M_PI*v, theta = M_PI*u;
		ans.x = flipped*cos(phi)*sin(theta);
		ans.y = flipped*sin(phi)*sin(theta);
		ans.z = flipped*cos(theta);
	}
	return ans;
}
double getObjArea(const sObject obj){
	if (obj.m_type==1)
		return myabs((obj.x0 - obj.x1)*(obj.y0 - obj.y1));
	else if (obj.m_type==2)
		return myabs((obj.x0 - obj.x1)*(obj.z0 - obj.z1));
	else if (obj.m_type==3)
		return myabs((obj.y0 - obj.y1)*(obj.z0 - obj.z1));
	else if (obj.m_type==5)
		return 4*M_PI*obj.radius*obj.radius;
	return 0;
}
bool hit(const sObject obj, const Material m, const Ray r, double t0, double t1, HitRecord *rec) {  //combined hit function
	if (obj.m_type==1) {
		if (obj.radius!=0){
			sObject tobj = obj;
			double objangle = obj.radius;
			double angle = objangle*M_PI/180.0;
			double cos_theta = cos(angle), sin_theta = sin(angle);
			tobj.radius = 0;
			double3 origin = r.a;
			double3 direction = r.b;
			// origin.x = origin.x - obj.x1;
			// origin.z = origin.z - obj.z1;
			origin.x = cos_theta*origin.x - sin_theta*origin.z;
			origin.z = sin_theta*origin.x + cos_theta*origin.z;
			direction.x = cos_theta*direction.x - sin_theta*direction.z;
			direction.z = sin_theta*direction.x + cos_theta*direction.z;
			// origin.x = origin.x + obj.x1;
			// origin.z = origin.z + obj.z1;
			Ray rotated_r = (Ray) {origin, direction};
			if (!hit(tobj,m,rotated_r,t0,t1,rec)){
				return false;
			}
			double3 p = rec->p;
			double3 normal = rec->normal;
			p.x = cos_theta*p.x + sin_theta*p.z;
			p.z = -sin_theta*p.x + cos_theta*p.z;
			normal.x = cos_theta*normal.x + sin_theta*normal.z;
			normal.z = -sin_theta*normal.x + cos_theta*normal.z;
			rec->p = p;
			rec->normal = normal;
			return true;
		}
		double t = (obj.k - r.a.z)/r.b.z;
		if (t<t0 || t>t1)
			return false;
		double3 p_at = PointAtParameter(r,t);
		double x = p_at.x;
		double y = p_at.y;
		if (x < obj.x0 || x > obj.x1 || y < obj.y0 || y > obj.y1)
        	return false;
		double2 uv = get_obj_uv(obj,p_at);
		rec->u = uv.x;
		rec->v = uv.y;
		rec->t = t;
		set_face_normal(r,(double3){0,0,1},rec);
		rec->m_curmat = m;
		rec->p = p_at;
		return true;
	}
	else if (obj.m_type==2){
		double t = (obj.k - r.a.y)/r.b.y;
		if (t<t0 || t>t1)
			return false;
		double3 p_at = PointAtParameter(r,t);
		double x = p_at.x;
		double z = p_at.z;
		if (x < obj.x0 || x > obj.x1 || z < obj.z0 || z > obj.z1)
        	return false;
		double2 uv = get_obj_uv(obj,p_at);
		rec->u = uv.x;
		rec->v = uv.y;
		rec->t = t;
		set_face_normal(r,(double3){0,1,0},rec);
		rec->m_curmat = m;
		rec->p = p_at;
		return true;
	}
	else if (obj.m_type==3){
		double t = (obj.k - r.a.x)/r.b.x;
		if (t<t0 || t>t1)
			return false;
		double3 p_at = PointAtParameter(r,t);
		double y = p_at.y;
		double z = p_at.z;
		if (y < obj.y0 || y > obj.y1 || z < obj.z0 || z > obj.z1)
        	return false;
		double2 uv = get_obj_uv(obj,p_at);
		rec->u = uv.x;
		rec->v = uv.y;
		rec->t = t;
		set_face_normal(r,(double3){1,0,0},rec);
		rec->m_curmat = m;
		rec->p = p_at;
		return true;
	}
	else if (obj.m_type==5){
		double3 center = (double3){obj.x0,obj.y0,obj.z0};
		double3 oc = r.a-center;
		double a = SquaredLength(r.b);
		double half_b = dot(oc,r.b);
		double c = SquaredLength(oc)-obj.radius*obj.radius;
		double discriminant = half_b*half_b - a*c;
		if (discriminant>0){
			double root = sqrt(discriminant);
			double temp = (-half_b-root)/a;
			bool poss=false;
			if (temp<t1 && temp>t0)
				poss=true;
			else{
				temp = (-half_b + root) / a;
				poss=temp < t1 && temp > t0;
			}
			if (poss){
				rec->t = temp;
				rec->p = PointAtParameter(r,temp);
				double3 outward_normal=(rec->p-center)/obj.radius;
				set_face_normal(r,outward_normal,rec);
				double2 uv = get_sphere_uv(outward_normal);
				rec->u = uv.x;
				rec->v = uv.y;
				rec->m_curmat = m;
				return true;
			}
			return false;
		}
		return false;
	}
	printf("Hihit\n");
	return false;
}
bool worldHit(const sObject *x, const Material *m, int ObjLen, const Ray r, HitRecord *rec) {
	HitRecord temp_rec;
	bool hitAnything = false;
	double closestSoFar = DBL_MAX;
	for (int i = 0; i < ObjLen; i++) {
		if (hit(x[i], m[i], r, 0.001, closestSoFar, &temp_rec)) {
			hitAnything = true;
			closestSoFar = temp_rec.t;
			*rec = temp_rec;
			rec->objId = i;
		}
	}
	return hitAnything;
}
double dydx(double* x, double* dxdt, double t, double rs){
	double3 current = (double3){x[0],x[1],x[2]};
	double R = length(current);
	double3 Xvec = (double3){x[3],x[4],x[5]};
	double pR = length(Xvec);
	double rs_4R = rs/(4*R);

	double inv_plus = 1/(1.0+rs_4R);
	double inv_plus_sq = inv_plus*inv_plus;
	double inv_plus_fr = inv_plus_sq*inv_plus_sq;
	double xcoeff = 0.0;
	if (rs < R)
		xcoeff = inv_plus_sq*inv_plus_fr*(1.0-rs_4R)*(1.0-rs_4R);
	
	//updating the position
	dxdt[0] = xcoeff*x[3];
	dxdt[1] = xcoeff*x[4];
	dxdt[2] = xcoeff*x[5];

	double inv_min_sq = 1/(1-rs_4R*rs_4R);
	double pcoeff = (-inv_min_sq - inv_plus_sq*inv_plus_fr*inv_plus*pR)*((1-rs_4R)*(1-rs_4R))*(rs_4R*2/(R*R));

	//updating the direction
	dxdt[3] = pcoeff*x[0];
	dxdt[4] = pcoeff*x[1];
	dxdt[5] = pcoeff*x[2];

	double3 rdasht = (double3){dxdt[0],dxdt[1],dxdt[2]};
	double3 rddasht = (double3){dxdt[3],dxdt[4],dxdt[5]};
	double lenrdasht = length(rdasht);
	double curvature = length(cross(rdasht,rddasht))/(lenrdasht*lenrdasht*lenrdasht);
	if (R <= rs)
		return -1;
	return curvature;
}
bool NonLinearWorldHit(const sObject *world, const Material *mats, int ObjLen, Ray *givenRay, HitRecord *rec, double* bbox) {
	double rs = 25*scale;
	double EPS = 1*scale;
	Ray r = *givenRay;
	r.b = normalize(givenRay->b);
	int gid = get_global_id(0);
	double boxFactor = length((double3){bbox[1]-bbox[0],bbox[3]-bbox[2],bbox[5]-bbox[4]});

	double curr_state[6];
	curr_state[0] = r.a.x;
	curr_state[1] = r.a.y;
	curr_state[2] = r.a.z;
	curr_state[3] = r.b.x;
	curr_state[4] = r.b.y;
	curr_state[5] = r.b.z;

	bool hasEntered = true;
	for (int i=0;i<6;i++){
		if (i%2==0 && curr_state[i/2] <= bbox[i]){
			hasEntered = false;
			break;
		}
		else if (i%2==1 && curr_state[i/2] >= bbox[i]){
			hasEntered = false;
			break;
		}
	}

	double dt, t=0, baseDt = 0.02*boxFactor;
	int iterations = 55;
	double prev_state[6];
	int maxNumReductions = 100, numReductions = 0;
	for(int i=0;i<iterations;i++){
		bool isOutside = false;
		for (int i=0;i<6;i++){
			if (i%2==0 && curr_state[i/2] <= bbox[i]){
				isOutside = true;
				break;
			}
			else if (i%2==1 && curr_state[i/2] >= bbox[i]){
				isOutside = true;
				break;
			}
		}
		if (isOutside){
			if (hasEntered)
				break;
		}
		else
			hasEntered = true;
		for (int j=0;j<6;j++)
			prev_state[j] = curr_state[j];

		//range kutta 4 do_step
		double dxdt[6];
		double k1vals[6], k2vals[6], k3vals[6], k4vals[6];
		// k1 = h*dydx(x, y)
		double curvature = boxFactor*dydx(curr_state, dxdt, t, rs);
		if (curvature<0)
			break;
		double redFac = min(max(sqrt(curvature),0.2),2.0);
		dt = baseDt / redFac;
		// dt = baseDt;
		// double currDis = length((double3){curr_state[0],curr_state[1],curr_state[2]});
		// if (currDis < 1000*scale && currDis > 11*scale)
		// 	printf("%f %f %f\n",currDis,curvature, dt);
		
		double tmp_state[6];
		for (int j=0;j<6;j++){
			k1vals[j] = dt * dxdt[j];
			tmp_state[j] = curr_state[j] + 0.5*k1vals[j];
		}
		// k2 = h*dydx(x + 0.5*h, y + 0.5*k1)
		dydx(tmp_state, dxdt, t + 0.5*dt, rs);
		for (int j=0;j<6;j++){
			k2vals[j] = dt * dxdt[j];
			tmp_state[j] = curr_state[j] + 0.5*k2vals[j];
		}
		//k3 = h*dydx(x + 0.5*h, y + 0.5*k2);
		dydx(tmp_state, dxdt, t + 0.5*dt, rs);
		for (int j=0;j<6;j++){
			k3vals[j] = dt * dxdt[j];
			tmp_state[j] = curr_state[j] + k3vals[j];
		}
		//k4 = h*dydx(x + h, y + k3);
		dydx(tmp_state, dxdt, t + dt, rs);
		for (int j=0;j<6;j++){
			k4vals[j] = dt * dxdt[j];
			curr_state[j] = curr_state[j] + (1.0/6.0)*(k1vals[j] + 2*k2vals[j] + 2*k3vals[j] + k4vals[j]);
		}
		// end rk4
		// curr_state[0] = curr_state[0] + dt*curr_state[3];
		// curr_state[1] = curr_state[1] + dt*curr_state[4];
		// curr_state[2] = curr_state[2] + dt*curr_state[5];
		
		double3 vc1 = (double3){prev_state[0],prev_state[1],prev_state[2]};
		double3 vc2 = (double3){curr_state[0],curr_state[1],curr_state[2]};
		Ray tmpRay = (Ray){vc1, normalize(vc2-vc1)};
		HitRecord temp_rec;
		bool foundHit = false, reducedt = false;
		double closestSoFar = dt;
		for (int objIdx = 0; objIdx < ObjLen; objIdx++){
			if (hit(world[objIdx], mats[objIdx], tmpRay, 0.0001, closestSoFar, &temp_rec)){
				double normalDist = length(temp_rec.p - vc1) * myabs(dot(temp_rec.normal, tmpRay.b));
				if (normalDist < EPS){
					foundHit = true;
					reducedt = false;
					closestSoFar = temp_rec.t;
					*rec = temp_rec;
					rec->objId = objIdx;
				}
				else if (!foundHit){
					reducedt = true;
				}
			}
		}
		// for (int objIdx = 0; objIdx < ObjLen; objIdx++){
		// 	bool hitCheck = hit(world[objIdx], mats[objIdx], tmpRay, 0.0001, closestSoFar, &temp_rec);
		// 	if (hitCheck){
		// 		// printf("Hi\n");
		// 		reducedt = true;
		// 		break;
		// 	}
		// 	else {
		// 		double normalDist = length(temp_rec.p - vc2) * myabs(dot(temp_rec.normal, tmpRay.b));
		// 		if (normalDist < EPS){
		// 			foundHit = true;
		// 			closestSoFar = temp_rec.t;
		// 			*rec = temp_rec;
		// 			rec->objId = objIdx;
		// 			break;
		// 		}
		// 	}
		// }
		if (foundHit){
			givenRay->a = vc1;
			givenRay->b = normalize((double3){prev_state[3],prev_state[4],prev_state[5]});
			return true;
		}
		else if (reducedt){
			for (int j=0;j<6;j++)
				curr_state[j] = prev_state[j];
			baseDt = baseDt/2;
			numReductions = numReductions + 1;
			// if (numReductions < maxNumReductions)
			// 	i = i - 1;
		}
		else
			t = t+dt;
	}
	return false;
}
double3 getEmitted(Material *mat,Ray r,HitRecord rec){  //emitted only for light material
	if (mat->m_MType!=1){
		return (double3){0,0,0};
	}
	else{
		bool cond = rec.front_face;
		if (mat->flip_face)
			cond = !rec.front_face;
		if (cond){
			return mat->m_vColor;
		}
		else{
			return (double3){0,0,0};
		}
	}
	printf("HiE\n");
	return (double3){0,0,0};
}
double double3_max(double3 val){  //max over R,G,B
	double ans = val.x;
	if (val.y>ans) 
		ans=val.y;
	if (val.z>ans) 
		ans=val.z;
	return ans;
}
double scattering_pdf(Material *mat, Ray r_in, HitRecord rec,Ray scattered){  //cosine pdf for scattering
	if (mat->m_MType == 0) {
		double cosine = dot(rec.normal,UnitVector(scattered.b));
		if (cosine<0)
			return 0;
		else
			return cosine/M_PI;
	}
	else if (mat->m_MType==1){
		return 0;
	}
	printf("HiSpdf\n");
	return 0;
}
double3 Color(const Ray *start,sObject *world, Material* mats, const int ObjLen, int max_depth, uint2* seed_ptr, global const double2* hammerselyPoints,global double* qtable, double* bbox) {
	double3 ans = (double3){1,1,1}, lastAttenuation;
	Ray r = *start;
	int idxQ = 0,prevOffset, nA = NUM_ACTIONS;
	for (int depth=0;depth<=max_depth;depth++){
		if (depth >= max_depth){
			ans = ans*(double3){0,0,0};
			break;
		}
		HitRecord rec;
		bool worldHitOutput;
		if (NON_LINEAR_ON==1)
			worldHitOutput = NonLinearWorldHit(world,mats,ObjLen,&r,&rec,bbox);
		else
			worldHitOutput = worldHit(world,mats,ObjLen,r,&rec);
		if (!worldHitOutput){
			ans = ans * background;
			break;
		}
		Ray scattered;
		double3 emitted = getEmitted(&rec.m_curmat,r,rec);
		int idxCurr=0,currOff=0;
		double Sum=0;
		if (RL_ON==1){
			if (USE_HAMM==1)
				idxCurr = getClosestIndexHamm(hammerselyPoints,world[rec.objId],rec.p);
			else
				idxCurr = getClosestIndexGrid(world[rec.objId],rec.p);
			currOff = (world[rec.objId].off + idxCurr)*nA;
			if (depth>0){
				double3 update;
				if (rec.m_curmat.m_MType==1)
					update = emitted;
				else{
					int qmaxid = currOff;
					for (int i=currOff;i<currOff+nA;i++){
						if (qtable[i]>qtable[qmaxid])
							qmaxid = i;
					}
					update = lastAttenuation*qtable[qmaxid];
				}
				//This region is for experimenting with Expected Sarsa
				// else {
				// 	int offset = currOff;
				// 	double sval=0,sumval=0;
				// 	for (int i=offset;i<offset+nA;i++) {
				// 		double3 diri = get_direction(rec.normal,i-offset,seed_ptr,true);
				// 		double brdf = dot(rec.normal,diri);
				// 		sval = sval + brdf*qtable[i];
				// 		sumval = sumval + brdf;
				// 	}
				// 	update = (sval/sumval)*lastAttenuation;
				// }
				double lr = 0.2;
				double update_val = double3_max(update);
				qtable[prevOffset] = (1-lr)*qtable[prevOffset] + lr*update_val;
			}
			double scatter_cdf[128];
			for (int i=0;i<nA;i++){
				Sum=Sum+qtable[i+currOff];
				scatter_cdf[i]=Sum;
			}
			double temprand = Sum*myrand(seed_ptr);
			idxQ = 0;
			while(idxQ<nA){
				if (temprand<=scatter_cdf[idxQ] || idxQ==nA)
					break;
				idxQ = idxQ + 1;
			}
		}
		
		double3 albedo;
		double pdf_val;
		if (!scatter(&rec.m_curmat,r,&rec,&albedo,&pdf_val,&scattered, seed_ptr, idxQ)){
			if (depth==0)
				emitted = emitted / double3_max(emitted);
			ans = ans * emitted;
			break;
		}
		if (RL_ON==1)
			pdf_val = (qtable[currOff+idxQ]/Sum)*nA/(2*M_PI);
		
		double3 attenuation;
		if (rec.m_curmat.m_MType==2 || rec.m_curmat.m_MType==3){
			attenuation = albedo;
			pdf_val = 1;
		}
		else{
			double spdf = scattering_pdf(&rec.m_curmat,r,rec,scattered);
			attenuation = albedo * spdf;
		}
		ans = ans * attenuation / pdf_val;
		r = scattered;
		if (RL_ON==1){
			lastAttenuation = attenuation;
			prevOffset = currOff + idxQ;
		}
	}
	return ans;
}
double3 squareToCosineHemisphere(uint2* seed_ptr){
	double phi, radius;
	double2 discSample;
	double remappedX = 2*myrand(seed_ptr) - 1;
	double remappedY = 2*myrand(seed_ptr) - 1;
	if (remappedX == 0 && remappedY == 0)
		discSample = (double2){0,0};
	else{
		if (remappedX*remappedX > remappedY*remappedY){
			radius = remappedX;
			phi = (M_PI/4)*(remappedY/remappedX);
		}
		else {
			radius = remappedY;
			phi = M_PI/2 - (M_PI/4)*(remappedX/remappedY);
		}
		discSample = (double2){radius*cos(phi),radius*sin(phi)};
	}
	double z = 1-SquaredLength2(discSample);
	z = sqrt(max(0.0,z));
	return (double3){discSample.x,discSample.y,z};
}
bool visibilityQuery(double3 start, double3 end, sObject *world, Material* mats, const int ObjLen){
	double3 joinDir = end-start;
	double dist = length(joinDir);
	joinDir = joinDir / dist;
	Ray joinRay = (Ray){start,joinDir};
	HitRecord dummyRec;
	bool hitAnything = false;
	double closestSoFar = DBL_MAX;
	double maxtval = dist-0.0001;
	for (int visidx=0;visidx<ObjLen;visidx++){
		if(hit(world[visidx], mats[visidx], joinRay, 0.0001, closestSoFar, &dummyRec)){
			closestSoFar = dummyRec.t;
			if (dummyRec.t>=0.0001 && dummyRec.t<=maxtval){
				hitAnything = true;
				break;
			}
		}
	}
	return hitAnything;
}

double FresnelDielectric(double etai, double etat, double cosi, double cost) {
	double eta = etai / etat;
	double sin2t = eta * eta * max(0.0, 1-cosi*cosi);
	if (sin2t >= 1.0)
		return 1.0;
	double rParallel = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	double rPerpendicular = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (rParallel * rParallel + rPerpendicular * rPerpendicular) * 0.5;
}

int2 splatToImagePlane (double3 position, int2 imageDims, Camera* camx){
	double aspectRatio = ((double)imageDims.x)/imageDims.y;
	Camera cam = camx[0];
	double worldToCamera[4][4];

	//start glmLookAt
	double3 f = UnitVector(cam.lookAt-cam.lookFrom);
	double3 s = UnitVector(cross(f, cam.viewUp));
	double3 u = cross(s,f);
	worldToCamera[0][0] = s.x;
	worldToCamera[1][0] = s.y;
	worldToCamera[2][0] = s.z;
	worldToCamera[0][1] = u.x;
	worldToCamera[1][1] = u.y;
	worldToCamera[2][1] = u.z;
	worldToCamera[0][2] = -f.x;
	worldToCamera[1][2] = -f.y;
	worldToCamera[2][2] = -f.z;
	worldToCamera[3][0] = -dot(s, cam.lookFrom);
	worldToCamera[3][1] = -dot(u, cam.lookFrom);
	worldToCamera[3][2] = dot(f, cam.lookFrom);
	worldToCamera[0][3] = 0.0;
	worldToCamera[1][3] = 0.0;
	worldToCamera[2][3] = 0.0;
	worldToCamera[3][3] = 1.0;
	//end glmLookAt
	// printf("WorldToCamera :\n");
	// for (int i=0;i<4;i++){
	// 	for (int j=0;j<4;j++)
	// 		printf("%f ",worldToCamera[i][j]);
	// 	printf("\n");
	// }
	
	double cameraToClip[4][4];
	//start glmPerspective
	double fovy = M_PI * cam.Fov / 180;
	double zNear = 1.0, zFar = 5000.0;
	double tanHalfFovy = tan(fovy / 2);
	for (int i=0;i<4;i++){
		for (int j=0;j<4;j++)
			cameraToClip[i][j] = 0.0;
	}
	cameraToClip[0][0] = 1.0 / (aspectRatio * tanHalfFovy);
	cameraToClip[1][1] = 1.0 / tanHalfFovy;
	cameraToClip[2][2] = - (zFar + zNear) / (zFar - zNear);
	cameraToClip[2][3] = -1.0;
	cameraToClip[3][2] = -2 * zFar * zNear / (zFar - zNear);
	//end glmPerspective
	// printf("cameraToClip :\n");
	// for (int i=0;i<4;i++){
	// 	for (int j=0;j<4;j++)
	// 		printf("%f ",cameraToClip[i][j]);
	// 	printf("\n");
	// }

	double NDCToScreen[4][4];

	for (int i=0;i<4;i++){
		for (int j=0;j<4;j++)
			NDCToScreen[i][j] = 0.0;
	}
	NDCToScreen[0][0] = imageDims.x * 0.5;
	NDCToScreen[1][1] = -imageDims.y * 0.5;
	NDCToScreen[2][2] = 1.0;
	NDCToScreen[3][3] = 1.0;
	NDCToScreen[3][0] = imageDims.x * 0.5;
	NDCToScreen[3][1] = imageDims.y * 0.5;
	NDCToScreen[3][2] = 0.0;

	// printf("NDCToScreen :\n");
	// for (int i=0;i<4;i++){
	// 	for (int j=0;j<4;j++)
	// 		printf("%f ",NDCToScreen[i][j]);
	// 	printf("\n");
	// }
	
	double tmpVector[4];
	tmpVector[0] = position.x;
	tmpVector[1] = position.y;
	tmpVector[2] = position.z;
	tmpVector[3] = 1.0;
	double uv[4];
	//uv = worldToCamera * v4f(position, 1.f)
	for (int i=0;i<4;i++){
		uv[i] = 0;
		for (int k=0;k<4;k++)
			uv[i] = uv[i] + worldToCamera[k][i] * tmpVector[k];
	}
	//uv = cameraToClip * uv
	for (int i=0;i<4;i++){
		tmpVector[i] = 0;
		for (int k=0;k<4;k++)
			tmpVector[i] = tmpVector[i] + cameraToClip[k][i] * uv[k];
	}
	//uv /= uv.w
	for (int i=0; i<4; i++)
		tmpVector[i] = tmpVector[i] / tmpVector[3];
	//uv = NDCToScreen * uv
	for (int i=0;i<4;i++){
		uv[i] = 0;
		for (int k=0;k<4;k++)
			uv[i] = uv[i] + NDCToScreen[k][i] * tmpVector[k];
	}
	// printf("vec1 :\n");
	// for (int i=0;i<4;i++){
	// 	printf("%f ",uv[i]);
	// }
	// printf("\n");
	int2 xyPixel = (int2){(int)uv[0],(int)uv[1]};
	// printf("Pixel: %d %d\n",xyPixel.x,xyPixel.y);
	return xyPixel;
}
double3 BiColor(const Ray *start,sObject *world, Material* mats, const int ObjLen, int max_depth, uint2* seed_ptr, Camera* camx, int2 imageDims, int numLights, int* lightObjIds, global double4 *pixel, double* bbox) {
	Camera cam = camx[0];
	PathVertex lightVertices[20];
	int lightDepth = 0;
	double3 totalAtten = (double3){0,0,0};
	int currLightIdx = 0;
	
	double connectToLightBoost = 1;
	double connectToCameraBoost = 1;
	double connectToVertexBoost = 1;
	double emittedBoost = 0.5;
	double throughputBoost = 1;
	//light subpath walk
	int emitterId = lightObjIds[currLightIdx];
	//sample emitter position
	double lightEmitteru = myrand(seed_ptr);
	double lightEmitterv = myrand(seed_ptr);
	double3 startPoint = getPointAtuv(lightEmitteru,lightEmitterv,world[emitterId]);
	double3 startnormal = getNormalAtuv(lightEmitteru,lightEmitterv,world[emitterId],mats[emitterId]);
	
	double3 emitterDirection = random_cosine_direction(seed_ptr); //handle for RL case
	// emitterDirection.z = max(emitterDirection.z,0.0001);
	double emitterPdf = 1.0 / numLights; //handle for multiple lights
	double emitterAreaPdf = (1.0/getObjArea(world[emitterId]));
	double emitterEmissionPdf = emitterAreaPdf/(2*M_PI);
	emitterAreaPdf = emitterAreaPdf * emitterPdf;
	emitterEmissionPdf = emitterEmissionPdf * emitterPdf;
	
	//get Start Ray
	Axes tmpaxes = getAxes(startnormal);
	double3 startDirection = emitterDirection.x*tmpaxes.s + emitterDirection.y*tmpaxes.t + emitterDirection.z*tmpaxes.n;
	Ray wi = (Ray){startPoint, startDirection};
	
	double3 throughput = throughputBoost * emitterDirection.z *mats[emitterId].m_vColor / emitterEmissionPdf;
	double vc = emitterDirection.z / emitterEmissionPdf;
	double vcm = emitterAreaPdf / emitterEmissionPdf;

	int depth = 0;
	while (depth<max_depth){
		HitRecord rec;
		bool worldHitOutput;
		if (NON_LINEAR_ON==1)
			worldHitOutput = NonLinearWorldHit(world,mats,ObjLen,&wi,&rec,bbox);
		else
			worldHitOutput = worldHit(world,mats,ObjLen,wi,&rec);
		if (!worldHitOutput){
			// ans = ans * background;
			break;
		}
		
		Axes pointAxes = getAxes(UnitVector(rec.normal));
		double3 hitWo = (double3){dot(pointAxes.s,-wi.b), dot(pointAxes.t,-wi.b), dot(pointAxes.n,-wi.b)};
		double distSquared = rec.t * rec.t;
		double absCosIn = myabs(hitWo.z);

		vcm = vcm * (distSquared / absCosIn);
		vc = vc / absCosIn;

		PathVertex lightVertex;
		lightVertex.rec = rec;
		lightVertex.throughput = throughput;
		lightVertex.vcm = vcm;
		lightVertex.vc = vc;
		lightVertex.wo = hitWo;
		
		bool isDeltaBSDF = rec.m_curmat.m_MType == 2 || rec.m_curmat.m_MType == 3 || rec.m_curmat.m_MType == 1;
		// start connectToCamera
		bool doConnectToCamera = NON_LINEAR_ON==0;
		if (!isDeltaBSDF && doConnectToCamera){
			double3 cameraForward = UnitVector(cam.lookAt - cam.lookFrom);
			double3 eyeToLightVert = lightVertex.rec.p - cam.lookFrom;
			double invDistanceSquared = 1.0 / SquaredLength(eyeToLightVert);
			eyeToLightVert = eyeToLightVert * sqrt(invDistanceSquared);

			int2 xyPixel = splatToImagePlane(rec.p,imageDims,camx);
			int xPixel = xyPixel.x;
			int yPixel = imageDims.y-1-xyPixel.y;
			// bool isDoor = world[rec.objId].m_type == 1 && world[rec.objId].radius!=0;
			double cosCamera = dot(cameraForward,eyeToLightVert);
			double sideCheck = dot(rec.normal, -eyeToLightVert);
			bool endConnectToCamera = xPixel < 0 || yPixel < 0 || xPixel >= imageDims.x || yPixel >= imageDims.y || cosCamera <= 0 || sideCheck <= 0;
			if (!endConnectToCamera){
				double3 bsdfevalIntrWi = (double3){dot(pointAxes.s,-eyeToLightVert), dot(pointAxes.t,-eyeToLightVert), dot(pointAxes.n,-eyeToLightVert)};
				if (bsdfevalIntrWi.z >=0 && hitWo.z>=0){
					double3 bsdfcosTheta = getObjColor(lightVertex.rec) * bsdfevalIntrWi.z/M_PI;
				
					if (!visibilityQuery(cam.lookFrom,rec.p,world,mats,ObjLen)) {
						double virtualNearPlaneDistance = imageDims.y * 0.5 / tan(cam.Fov * M_PI/360);
						double imagePointToCameraDist = virtualNearPlaneDistance / cosCamera;
						double imageAreaToCameraSolidAngle = imagePointToCameraDist * imagePointToCameraDist / cosCamera;
						double cameraSolidAngleToSurfaceArea = bsdfevalIntrWi.z * invDistanceSquared;
						double imageAreaToSurfaceArea = imageAreaToCameraSolidAngle * cameraSolidAngleToSurfaceArea;
						double surfaceAreaToImageArea = 1.0 / imageAreaToSurfaceArea;
						
						int nlightSubpath = imageDims.x * imageDims.y;
						double3 radiance = connectToCameraBoost * lightVertex.throughput*bsdfcosTheta/(bsdfevalIntrWi.z*surfaceAreaToImageArea*nlightSubpath);
						double reversePdf_a = 1.0 * imageAreaToSurfaceArea;
						double previousVertexReversePdf_w = max(0.0,hitWo.z) / M_PI;
						double lightWeight = (reversePdf_a / nlightSubpath) * (lightVertex.vcm + previousVertexReversePdf_w * lightVertex.vc);
						double misWeight = 1.0 / (1.0 + lightWeight);

						radiance = misWeight * radiance;
						int imagePixel = yPixel * imageDims.x + xPixel;
						pixel[imagePixel] = pixel[imagePixel] + (double4){radiance.x,radiance.y,radiance.z,0};
						// pixel[imagePixel] = pixel[imagePixel] + (double4){35,35,35,0};
					}
				}
			}
		}
		// end connectToCamera
		//start ContinuePathRandomWalk
		double bsdfPdf_w;
		double3 localWi, bsdfcosTheta;
		bool continuePath = false;
		double3 objColor = getObjColor(rec);
		
		if (rec.m_curmat.m_MType == 2){
			localWi = (double3){-hitWo.x,-hitWo.y,hitWo.z};
			bsdfPdf_w = 1.0;
			bsdfcosTheta = objColor;
			continuePath = true;
		}
		else if (rec.m_curmat.m_MType == 3){
			bsdfPdf_w = 1.0;
			double ref_idx = 1.5;

			bool isEntering = hitWo.z > 0;
			double etai = 1.0, etat = ref_idx;
			if (!isEntering){
				etai = ref_idx;
				etat = 1.0;
			}
			double eta = etai / etat;
			double sin2i = max(0.0, 1 - hitWo.z*hitWo.z);
			double sin2t = eta * eta * sin2i;
			double cost = sqrt(max(0.0,1 - sin2t));
			if (isEntering)
				cost = -cost;

			// double fresnel = FresnelDielectric(etai, etat, myabs(hitWo.z), myabs(cost)); 
			double reflect_prob = schlick(hitWo.z, eta);

			if (eta * sqrt(sin2i) > 1.0 || myrand(seed_ptr) < reflect_prob){
				bsdfcosTheta = objColor;
				localWi = (double3){-hitWo.x,-hitWo.y,hitWo.z};
			}
			else {
				bsdfcosTheta = objColor;
				localWi = UnitVector((double3){-eta * hitWo.x, -eta * hitWo.y, cost});
				// double3 transmitted = Refract(UnitVector(wi.b),rec.normal,eta);
				// wi.b = transmitted;
				// localWi = (double3){dot(transmitted, pointAxes.s), dot(transmitted, pointAxes.t), dot(transmitted, pointAxes.n)};
			}
			continuePath = true;
		}
		else if (rec.m_curmat.m_MType==1)
			continuePath = false;
		else {
			localWi = squareToCosineHemisphere(seed_ptr);
			bsdfPdf_w = max(0.0,localWi.z)/M_PI;
			bsdfcosTheta = objColor * localWi.z / M_PI;
			continuePath = true;
		}
		// lightVertex.wi = localWi;
		if (continuePath){
			double absCosOut = myabs(localWi.z);
			throughput = throughput * objColor;
			depth = depth + 1;
			double previousVertexReversePdf_w;
			if (isDeltaBSDF)
				previousVertexReversePdf_w = bsdfPdf_w;
			else
				previousVertexReversePdf_w = max(0.0,hitWo.z)/M_PI;
			if (isDeltaBSDF){
				vc = (absCosOut / bsdfPdf_w) * previousVertexReversePdf_w * vc;
				// vc = M_PI * previousVertexReversePdf_w * vc;
				vcm = 0;
			}
			else{
				vc = (absCosOut / bsdfPdf_w) * (vcm + previousVertexReversePdf_w * vc);
				// vc = M_PI * (vcm + previousVertexReversePdf_w * vc);
				vcm = 1.0 / bsdfPdf_w;
			}
			if (rec.m_curmat.m_MType == 3){
				double ref_idx = 1.5;
				double etai_over_etat = rec.front_face ? (1.0 / ref_idx) : ref_idx;
				double3 unit_direction = UnitVector(wi.b);
				double cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
				double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
				double reflect_prob = schlick(cos_theta, etai_over_etat);
				double3 scattered;
				if (etai_over_etat * sin_theta > 1.0 || myrand(seed_ptr) < reflect_prob) {
					scattered = Reflect(unit_direction, rec.normal);
				}
				else{
					scattered = Refract(unit_direction, rec.normal, etai_over_etat);
				}
				wi = (Ray){rec.p,scattered};
			}
			else 
				wi = (Ray){rec.p, pointAxes.s * localWi.x + pointAxes.t * localWi.y + pointAxes.n * localWi.z};
		}
		else
			break;
		//end ContinuePathRandomWalk
		if (!isDeltaBSDF){
			lightVertices[lightDepth] = lightVertex;
			lightDepth = lightDepth + 1;
		}
	}
	

	//eyeSubpathWalk
	Ray cameraStartRay = *start;
	bool isPathPureSpecular = false;
	double3 cameraForward = UnitVector(cam.lookAt - cam.lookFrom);
	double cosCamera = dot(cameraForward, cameraStartRay.b);
	double virtualNearPlaneDistance = imageDims.y * 0.5 / tan(cam.Fov * M_PI/360); //markedPos
	double imagePointToCameraDist = virtualNearPlaneDistance / cosCamera;
	double imageAreaToCameraSolidAngle = imagePointToCameraDist * imagePointToCameraDist / cosCamera;

	double t1Pdf = 1.0 * imageAreaToCameraSolidAngle;
	wi = cameraStartRay;
	throughput = (double3){1,1,1};
	vc = 0;
	vcm = imageDims.x * imageDims.y / t1Pdf;

	depth = 0;
	while (depth<max_depth){
		HitRecord rec;
		bool worldHitOutput;
		if (NON_LINEAR_ON==1)
			worldHitOutput = NonLinearWorldHit(world,mats,ObjLen,&wi,&rec,bbox);
		else
			worldHitOutput = worldHit(world,mats,ObjLen,wi,&rec);
		if (!worldHitOutput){
			break;
		}
		Axes pointAxes = getAxes(UnitVector(rec.normal));
		double3 hitWo = (double3){dot(pointAxes.s,-wi.b), dot(pointAxes.t,-wi.b), dot(pointAxes.n,-wi.b)};
		double distSquared = rec.t * rec.t;
		double absCosIn = myabs(hitWo.z);

		vcm = vcm * (distSquared / absCosIn);
		vc = vc / absCosIn;
		double3 emitted = getEmitted(&rec.m_curmat,wi,rec);
		if (rec.m_curmat.m_MType==1){
			double emitterPdf = 1.0 / numLights; //handle for multiple lights
			if (depth>0){
				double3 contribution = emittedBoost * emitted * throughput;
				double emitterPositionPdf_a = 1.0 / (getObjArea(world[rec.objId]) * emitterPdf);
				double emitterDirectionPdf_w = 1/(2*M_PI);

				double cameraWeight = emitterPositionPdf_a * vcm + (emitterPositionPdf_a * emitterDirectionPdf_w) * vc;
				double misWeight = 1.f / (1.f + cameraWeight);

				if (!isPathPureSpecular)
					contribution = contribution * misWeight;
				totalAtten = totalAtten + contribution;
				// printf("%f %f %f\n",contribution.x,contribution.y,contribution.z);
			}
			else
				totalAtten = totalAtten + emitted / double3_max(emitted);
			break;
		}
		PathVertex cameraVertex;
		cameraVertex.rec = rec;
		cameraVertex.throughput = throughput;
		cameraVertex.vcm = vcm;
		cameraVertex.vc = vc;
		cameraVertex.wo = hitWo;
		bool isDeltaBSDF1 = rec.m_curmat.m_MType==2 || rec.m_curmat.m_MType==3;
		if (!isDeltaBSDF1){
			isPathPureSpecular = false;

			//start connectToLight
			int emitterID = lightObjIds[currLightIdx];
			// currLightIdx = (1 + currLightIdx) % numLights;
			double emitterPdf = 1.0 / numLights;
			//sample emitter position
			double emitteru = myrand(seed_ptr);
			double emitterv = myrand(seed_ptr);
			double3 emitterPosition = getPointAtuv(emitteru,emitterv,world[emitterId]);
			double3 emitterNormal = getNormalAtuv(emitteru,emitterv,world[emitterId],mats[emitterId]);
			double emitterPositionPdf = 1.0 / getObjArea(world[emitterId]);

			double3 lightToEyeVertexDir = cameraVertex.rec.p - emitterPosition;
			double lightToEyeDistSqrd = SquaredLength(lightToEyeVertexDir);
			lightToEyeVertexDir = lightToEyeVertexDir / sqrt(lightToEyeDistSqrd);
			double3 bsdfevalIntrWi = (double3){dot(pointAxes.s,-lightToEyeVertexDir), dot(pointAxes.t,-lightToEyeVertexDir), dot(pointAxes.n,-lightToEyeVertexDir)};
			// cameraVertex.wi = bsdfevalIntrWi; // check this

			double cosAtLight = dot(emitterNormal,lightToEyeVertexDir);
			double cosAtEyeVertex = bsdfevalIntrWi.z;
			double sideCheck = dot(rec.normal, -lightToEyeVertexDir);

			if (cosAtLight > 0 && cosAtEyeVertex > 0 && sideCheck > 0){
				double emitterConnectPdf_a = emitterPdf * emitterPositionPdf;
				double emitterConnectPdf_w = emitterConnectPdf_a * lightToEyeDistSqrd / cosAtLight;
				double emitterDirectionPdf_w = 1/(2*M_PI);

				double3 Li = getObjColor(cameraVertex.rec) * (bsdfevalIntrWi.z/M_PI) * (cameraVertex.throughput * connectToLightBoost * mats[emitterId].m_vColor / emitterConnectPdf_w); //lightBoost
				//visibility Query
				if (hitWo.z>=0 && !visibilityQuery(cameraVertex.rec.p,emitterPosition,world,mats,ObjLen)){
					double lightPathReversePdf_w = max(0.0,bsdfevalIntrWi.z)/M_PI;
					double lightPathReversePdf_a = lightPathReversePdf_w * cosAtLight / lightToEyeDistSqrd;
					double lightWeight = lightPathReversePdf_w / emitterConnectPdf_w;
					// double lightWeight = lightPathReversePdf_a * 2*M_PI*(1 + lightPathReversePdf_w * (cosAtLight / emitterConnectPdf_a));
					
					double eyePathPreviousVertexReversePdf_w = max(0.0,hitWo.z)/M_PI;
					double eyePathCurrentVertexReversePdf_a = cosAtEyeVertex * emitterDirectionPdf_w / lightToEyeDistSqrd;
					double eyeWeight = eyePathCurrentVertexReversePdf_a * (cameraVertex.vcm + eyePathPreviousVertexReversePdf_w * cameraVertex.vc);
					double misWeight = 1.0 / (lightWeight + 1.0 + eyeWeight);

					// Li = Li * misWeight;
					totalAtten = totalAtten + Li;
				}
			}
			//end connectToLight
			//start Connect vertices
			for (int lvidx = 0; lvidx < lightDepth; lvidx++){
				PathVertex eyeVertex = cameraVertex;
				PathVertex lightVertex = lightVertices[lvidx];
				double3 lightToEyeDir = eyeVertex.rec.p - lightVertex.rec.p;
				double invDistanceSquared = 1.0 / SquaredLength(lightToEyeDir);
				lightToEyeDir = lightToEyeDir * sqrt(invDistanceSquared);

				Axes lightConnectAxes = getAxes(lightVertex.rec.normal);
				double3 lightVertexHitwi = (double3){dot(lightConnectAxes.s,lightToEyeDir),dot(lightConnectAxes.t,lightToEyeDir),dot(lightConnectAxes.n,lightToEyeDir)};
				Axes eyeConnectAxes = getAxes(eyeVertex.rec.normal);
				double3 eyeVertexHitwi = (double3){dot(eyeConnectAxes.s,-lightToEyeDir),dot(eyeConnectAxes.t,-lightToEyeDir),dot(eyeConnectAxes.n,-lightToEyeDir)};

				double cosAtLightVertex = lightVertexHitwi.z;
				double cosAtEyeVertex = eyeVertexHitwi.z;
				if (cosAtLightVertex>0 && cosAtEyeVertex>0 && !visibilityQuery(eyeVertex.rec.p,lightVertex.rec.p,world,mats,ObjLen)){
					double3 Li = (double3){0,0,0};
					if (lightVertexHitwi.z >=0 && lightVertex.wo.z>=0 && eyeVertexHitwi.z>=0 && eyeVertex.wo.z>=0){
						Li = getObjColor(lightVertex.rec) * (lightVertexHitwi.z / M_PI) * getObjColor(eyeVertex.rec) * (eyeVertexHitwi.z / M_PI);
						Li = Li * connectToVertexBoost * lightVertex.throughput * eyeVertex.throughput * invDistanceSquared;
						double eyePathReversePdf_w = max(0.0,lightVertexHitwi.z)/M_PI;
						double lightPathPreviousVertexReversePdf_w = max(0.0,lightVertex.wo.z)/M_PI;

						double lightPathReversePdf_w = max(0.0,eyeVertexHitwi.z)/M_PI;
						double eyePathPreviousVertexReversePdf_w = max(0.0,eyeVertex.wo.z)/M_PI;

						double lightPathReversePdf_a = lightPathReversePdf_w * cosAtLightVertex * invDistanceSquared;
						double eyePathReversePdf_a = eyePathReversePdf_w * cosAtEyeVertex * invDistanceSquared;

						double lightWeight = lightPathReversePdf_a * (lightVertex.vcm + lightPathPreviousVertexReversePdf_w * lightVertex.vc);
						double eyeWeight = eyePathReversePdf_a * (eyeVertex.vcm + eyePathPreviousVertexReversePdf_w * eyeVertex.vc);
						double misWeight = 1.f / (lightWeight + 1.f + eyeWeight);
						Li = Li * misWeight;
						totalAtten = totalAtten + Li;
					}
				}
			}
			//end Connect vertices
		}
		//start ContinuePathRandomWalk
		bool isDeltaBSDF = rec.m_curmat.m_MType == 2 || rec.m_curmat.m_MType == 3;
		double bsdfPdf_w;
		double3 localWi, bsdfcosTheta;
		bool continuePath = false;
		double3 objColor = getObjColor(rec);
		if (rec.m_curmat.m_MType == 2){
			localWi = (double3){-hitWo.x,-hitWo.y,hitWo.z};
			bsdfPdf_w = 1.0;
			bsdfcosTheta = objColor;
			continuePath = true;
		}
		else if (rec.m_curmat.m_MType == 3){
			bsdfPdf_w = 1.0;
			double ref_idx = 1.5;

			bool isEntering = hitWo.z > 0;
			double etai = 1.0, etat = ref_idx;
			if (!isEntering){
				etai = ref_idx;
				etat = 1.0;
			}
			double eta = etai / etat;
			double sin2i = max(0.0, 1 - hitWo.z*hitWo.z);
			double sin2t = eta * eta * sin2i;
			double cost = sqrt(max(0.0,1 - sin2t));
			if (isEntering)
				cost = -cost;

			double reflect_prob = schlick(hitWo.z, eta);

			if (eta * sqrt(sin2i) > 1.0 || myrand(seed_ptr) < reflect_prob){
				bsdfcosTheta = objColor;
				localWi = (double3){-hitWo.x,-hitWo.y,hitWo.z};
			}
			else {
				bsdfcosTheta = objColor;
				localWi = UnitVector((double3){-eta * hitWo.x, -eta * hitWo.y, cost});
				// double3 transmitted = Refract(UnitVector(wi.b),rec.normal,eta);
				// wi.b = transmitted;
				// localWi = (double3){dot(transmitted, pointAxes.s), dot(transmitted, pointAxes.t), dot(transmitted, pointAxes.n)};
			}
			continuePath = true;
		}
		else {
			localWi = squareToCosineHemisphere(seed_ptr);
			bsdfPdf_w = max(0.0,localWi.z)/M_PI;
			bsdfcosTheta = objColor * localWi.z / M_PI;
			continuePath = true;
		}
		if (continuePath){
			double absCosOut = myabs(localWi.z);
			throughput = throughput * bsdfcosTheta / bsdfPdf_w;
			depth = depth + 1;
			double previousVertexReversePdf_w;
			if (isDeltaBSDF)
				previousVertexReversePdf_w = bsdfPdf_w;
			else
				previousVertexReversePdf_w = max(0.0,hitWo.z)/M_PI;
			if (isDeltaBSDF){
				vc = (absCosOut / bsdfPdf_w) * previousVertexReversePdf_w * vc;
				// vc = M_PI * previousVertexReversePdf_w * vc;
				vcm = 0;
			}
			else{
				vc = (absCosOut / bsdfPdf_w) * (vcm + previousVertexReversePdf_w * vc);
				// vc = M_PI * (vcm + previousVertexReversePdf_w * vc);
				vcm = 1.0 / bsdfPdf_w;
			}
			if (rec.m_curmat.m_MType == 3){
				double ref_idx = 1.5;
				double etai_over_etat = rec.front_face ? (1.0 / ref_idx) : ref_idx;
				double3 unit_direction = UnitVector(wi.b);
				double cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
				double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
				double reflect_prob = schlick(cos_theta, etai_over_etat);
				double3 scattered;
				if (etai_over_etat * sin_theta > 1.0 || myrand(seed_ptr) < reflect_prob) {
					scattered = Reflect(unit_direction, rec.normal);
				}
				else{
					scattered = Refract(unit_direction, rec.normal, etai_over_etat);
				}
				wi = (Ray){rec.p,scattered};
			}
			else 
				wi = (Ray){rec.p, pointAxes.s * localWi.x + pointAxes.t * localWi.y + pointAxes.n * localWi.z};
		}
		else
			break;
		//end ContinuePathRandomWalk
	}
	// printf("%f %f %f\n",totalAtten.x,totalAtten.y,totalAtten.z);
	return totalAtten;
}
Ray getRay(double s, double t, int2 dims, Camera cam,uint2* seed_ptr) { //get Ray from pixel according to (s,t) in pixel

	double dHalfHeight = tan(cam.Fov*M_PI / 360);
	double dHalfWidth = ((double)dims.x / dims.y) * dHalfHeight;
	double dFocusDist = length(cam.lookFrom - cam.lookAt);
	// double dFocusDist = cam.focus_dist;
	double3 vW = UnitVector(cam.lookFrom - cam.lookAt);
	double3 vU = UnitVector(cross(cam.viewUp, vW));
	double3 vV = cross(vW, vU);
	double3 vOrigin = cam.lookFrom;
	double3 vLowerLeftCorner = vOrigin - (dHalfWidth * dFocusDist * vU) - (dHalfHeight * dFocusDist * vV) - (dFocusDist * vW);
	double3 vHorizontal = 2 * dHalfWidth*dFocusDist*vU;
	double3 vVertical = 2 * dHalfHeight*dFocusDist*vV;
	double3 vRD = (cam.aperture / 2) * random_in_unit_disk(seed_ptr);
	double3 vOffset = vU * vRD.x + vV * vRD.y;
	return (Ray) { (double3)(vOrigin + vOffset), normalize((double3)(vLowerLeftCorner + (s * vHorizontal) + (t * vVertical) - vOrigin - vOffset)) };
}

kernel void Render(global double4 *pixel, global int2 *dims, global const double16 *cam, global const double8 *list, global const int *listLen, global const double4 *materials, global const double2* hammerselyPoints,global const int* ssp, global const int* hamOff,global double* qtable) {

	int gid = get_global_id(0); // Current ray in image
	int raysPerPixel = ssp[0];
	int doRS = ssp[1];
	int gsize = get_global_size(0); // Number of rays in pixel
	int2 dim = dims[0]; // Image Dimensions
	int ObjLen = listLen[0]; // # objects in list
	int wgNum = get_group_id(0);
	
	int j = gid / dim.x;
	int i = gid % dim.x;
	// int j = floor((double)(1 + (gid / dim.x))); // Current Y
	// int i = gid - ((j - 1) * dim.x); // Current X
	// Object list initialized from kernel struct, max objects in image defined by array size
	sObject world[20];
	Material mats[20];
	Camera camx[1];

	double bbox[6];
	for (int i=0;i<6;i++){
		if (i%2==0)
			bbox[i] = DBL_MAX;
		else
			bbox[i] = DBL_MIN;
	}
	int tot_points=0;
	for (int i = 0; i < ObjLen; i++) {
		world[i].m_type = (int)list[i].s7;
		if (USE_HAMM==1){
			if (i==0){
				world[i].off = 0;
				world[i].numPoints = hamOff[i];
			}
			else{
				world[i].off = hamOff[i-1];
				world[i].numPoints = hamOff[i] - hamOff[i-1];
			}
		}
		else{
			world[i].off = tot_points;
			world[i].numPoints = hamOff[i];
			tot_points = tot_points + hamOff[i]*hamOff[i];
		}
		if (world[i].m_type == 1){
			world[i].x0 = list[i].s0;
			world[i].x1 = list[i].s1;
			world[i].y0 = list[i].s2;
			world[i].y1 = list[i].s3;
			world[i].z0 = world[i].z1 = world[i].k = list[i].s4;
			world[i].radius = list[i].s6;  //rotation amount stored in radius for rectangles
			bbox[0] = min(min(bbox[0],world[i].x0),world[i].x1);
			bbox[1] = max(max(bbox[1],world[i].x0),world[i].x1);
			bbox[2] = min(min(bbox[2],world[i].y0),world[i].y1);
			bbox[3] = max(max(bbox[3],world[i].y0),world[i].y1);
			bbox[4] = min(bbox[4],world[i].k);
			bbox[5] = max(bbox[5],world[i].k);
		}
		else if (world[i].m_type == 2){
			world[i].x0 = list[i].s0;
			world[i].x1 = list[i].s1;
			world[i].z0 = list[i].s2;
			world[i].z1 = list[i].s3;
			world[i].y0 = world[i].y1 = world[i].k = list[i].s4;
			world[i].radius = list[i].s6;
			bbox[0] = min(min(bbox[0],world[i].x0),world[i].x1);
			bbox[1] = max(max(bbox[1],world[i].x0),world[i].x1);
			bbox[4] = min(min(bbox[4],world[i].z0),world[i].z1);
			bbox[5] = max(max(bbox[5],world[i].z0),world[i].z1);
			bbox[2] = min(bbox[2],world[i].k);
			bbox[3] = max(bbox[3],world[i].k);
		}
		else if (world[i].m_type == 3){
			world[i].y0 = list[i].s0;
			world[i].y1 = list[i].s1;
			world[i].z0 = list[i].s2;
			world[i].z1 = list[i].s3;
			world[i].x0 = world[i].x1 = world[i].k = list[i].s4;
			world[i].radius = list[i].s6;
			bbox[4] = min(min(bbox[4],world[i].z0),world[i].z1);
			bbox[5] = max(max(bbox[5],world[i].z0),world[i].z1);
			bbox[2] = min(min(bbox[2],world[i].y0),world[i].y1);
			bbox[3] = max(max(bbox[3],world[i].y0),world[i].y1);
			bbox[0] = min(bbox[0],world[i].k);
			bbox[1] = max(bbox[1],world[i].k);
		}
		else if (world[i].m_type == 5){
			world[i].x0 = list[i].s0;
			world[i].y0 = list[i].s1;
			world[i].z0 = list[i].s2;
			world[i].radius = list[i].s3;
			bbox[0] = min(bbox[0],world[i].x0 - world[i].radius);
			bbox[1] = max(bbox[1],world[i].x0 + world[i].radius);
			bbox[2] = min(bbox[2],world[i].y0 - world[i].radius);
			bbox[3] = max(bbox[3],world[i].y0 + world[i].radius);
			bbox[4] = min(bbox[4],world[i].z0 - world[i].radius);
			bbox[5] = max(bbox[5],world[i].z0 + world[i].radius);
		}

		mats[i].m_vColor = (double3){ materials[i].s0, materials[i].s1, materials[i].s2 };
		mats[i].m_MType = (int)(materials[i].s3);
		mats[i].flip_face = i!=5 && mats[i].m_MType == 1 && world[i].m_type != 5;
	}
	for (int i=0;i<6;i++){
		if (i%2==0)
			bbox[i] = bbox[i] - 10;
		else
			bbox[i] = bbox[i] + 10;
	}

	camx[0].lookFrom = (double3) { cam[0].s0, cam[0].s1, cam[0].s2 };
	camx[0].lookAt = (double3) { cam[0].s3, cam[0].s4, cam[0].s5 };
	camx[0].viewUp = (double3) { cam[0].s6, cam[0].s7, cam[0].s8 };
	camx[0].Fov = cam[0].s9;
	camx[0].aperture = cam[0].sa;
	camx[0].focus_dist = cam[0].sb;
	
	double3 col = (double3)(0);

	int count[1];
	count[0]=0;
	int max_depth = MAX_DEPTH;
	uint2 SEED;
	SEED.x = gid;
	SEED.y = SEED.x+1;
	int iters = 1;
	int numLights = 0;
	int lightObjIds[10];
	for (int objIdx=0;objIdx<ObjLen;objIdx++){
		if (mats[objIdx].m_MType==1){
			lightObjIds[numLights] = objIdx;
			numLights = numLights + 1;
		}
	}
	if (RL_ON==1 && doRS==1 && gid<dim.x*dim.y/10)
		iters = 2 + NUM_POINTS/50;
		// iters = 1;
	for (int iter=0;iter<iters;iter++){
		col = (double3)(0);
		for (int s = 0; s < raysPerPixel; s++) {
			double u = (double)(i + myrand(&SEED)) / (double)dim.x;
			double v = (double)(j + myrand(&SEED)) / (double)dim.y;
			Ray r = getRay(u, v, dim, camx[0],&SEED);
			if (BIDIR_ON==1)
				col += BiColor(&r, world, mats, ObjLen, max_depth, &SEED,camx,dim,numLights,lightObjIds,pixel,bbox);
			else
				col += Color(&r, world, mats, ObjLen, max_depth, &SEED, hammerselyPoints,qtable,bbox);
		}
	}
	// col = sqrt(col / raysPerPixel);             //No need for gamma correction as it is done while writing to image file by CPU
	
	pixel[gid] = (double4)(col, gid);
	double numElements = dim.x*dim.y;
	// if (gid==0){
	// 	splatToImagePlane((double3){1,2,3}, dim, camx);
	// }
	printf("\r%.2f percent",100*gid/numElements);  //Print progress
}