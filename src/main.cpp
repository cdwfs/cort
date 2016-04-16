#include "cds_float3.h"
#include "platform.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if   defined(ZOMBO_COMPILER_MSVC)
#   if _MSC_VER < 1900
#       define CDS_THREADLOCAL __declspec(thread)
#   else
#       define CDS_THREADLOCAL thread_local
#   endif
#elif defined(ZOMBO_COMPILER_GNU)
#   define CDS_THREADLOCAL __thread
#elif defined(ZOMBO_COMPILER_CLANG)
#   if defined(ZOMBO_PLATFORM_APPLE)
#       define CDS_THREADLOCAL __thread
#   else
#       define CDS_THREADLOCAL thread_local
#   endif
#endif

#include <assert.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <float.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <thread>
#include <vector>

#if !defined(ZOMBO_COMPILER_MSVC)
#   include <algorithm>
using std::min;
#endif

#if !defined(M_PI)
#   define M_PI 3.14159265358979323846
#endif

static inline float clamp(float x, float a, float b)
{
    return (x<a) ? a : (x>b ? b : x);
}

template<typename T>
class AnimationChannel
{
public:
    AnimationChannel() = delete;
    struct KeyFrame
    {
        float time;
        T value;
    };
    AnimationChannel(const T &constantValue) // TODO(cort): make it explicit?
        :   m_keyTimes(1)
        ,   m_keyValues(1)
        ,   m_segmentMatrices(0)
    {
        m_keyTimes[0] = 0.0f;
        m_keyValues[0] = constantValue;
    }
    explicit AnimationChannel(const std::vector<KeyFrame> &keyFrames)
        :   m_keyTimes(keyFrames.size())
        ,   m_keyValues(keyFrames.size())
        ,   m_segmentMatrices(keyFrames.size() > 0 ? keyFrames.size()-1 : 0)
    {
        int i=0;
        for(const auto itor : keyFrames)
        {
            m_keyTimes[i] = itor.time;
            m_keyValues[i] = itor.value;
            ++i;
        }
        const float tau = 0.5f;
        for(int iSeg=0; iSeg<m_segmentMatrices.size(); ++iSeg)
        {
            const T p1 = m_keyValues[iSeg+0];
            const T p2 = m_keyValues[iSeg+1];
            const T p0 = (iSeg==0) ? p1 : m_keyValues[iSeg-1];
            const T p3 = (iSeg==m_segmentMatrices.size()-1) ? p2 : m_keyValues[iSeg+2];
            m_segmentMatrices[iSeg] = {
                p1,
                -tau*p0 + tau*p2,
                2*tau*p0 + (tau-3)*p1 + (3-2*tau)*p2 + -tau*p3,
                -tau*p0 + (2-tau)*p1 + (tau-2)*p2 + tau*p3,
            };
        }
    }

    T eval(float t) const
    {
        if (t <= m_keyTimes[0])
            return m_keyValues[0];
        if (t >= m_keyTimes[m_keyTimes.size()-1])
            return m_keyValues[m_keyTimes.size()-1];

        int segmentCount = (int)m_segmentMatrices.size();
        int firstSegment=0, lastSegment=segmentCount, iSegment;
        for(;;)
        {
            iSegment = (firstSegment+lastSegment)/2;
            if (t >= m_keyTimes[iSegment] && t<=m_keyTimes[iSegment+1])
                break;
            assert(firstSegment != lastSegment); // search will never terminate
            if (t < m_keyTimes[iSegment])
                lastSegment = iSegment;
            else // if (t > m_keyTimes[iSegment+1])
                firstSegment = iSegment;
        }
        float t01 = (t-m_keyTimes[iSegment]) / (m_keyTimes[iSegment+1]-m_keyTimes[iSegment]);
        const auto &seg = m_segmentMatrices[iSegment];
        return ((seg.m[3]*t01 + seg.m[2])*t01 + seg.m[1])*t01 + seg.m[0];
    }
private:
    struct SegmentMatrix
    {
        T m[4];
    };
    std::vector<float> m_keyTimes;
    std::vector<T> m_keyValues;
    std::vector<SegmentMatrix> m_segmentMatrices;
};

// Computes a refracted vector. Returns false if total internal reflection occurs.
// niOverNt is refractionIndex when entering an object, and 1/refractionIndex when exiting into air.
static CDSF3_INLINE bool CDSF3_VECTORCALL refract(float3 vIn, float3 n, float niOverNt, float3 *vOut)
{
    float3 vInUnit = normalize(vIn);
    float dt = dot(vInUnit, n);
    float discriminant = 1.0f - niOverNt*niOverNt*(1.0f-dt*dt);
    if (discriminant > 0)
    {
        *vOut = niOverNt * (vInUnit - n*dt) - n*sqrtf(discriminant);
        return true;
    }
    else
    {
        return false;
    }
}

class RNG
{
public:
    RNG()
        :   RNG( (unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count() )
    {}
    explicit RNG(unsigned long seed)
        :   m_randomGen(seed)
        ,   m_uniform(0.0f, 1.0f)
        ,   m_uniformUint32(0, UINT_MAX)
    {}

    ZOMBO_INLINE void seed(unsigned long s)
    {
        return m_randomGen.seed(s);
    }

    ZOMBO_INLINE float random01(void)
    {
        return m_uniform(m_randomGen);
    }

    ZOMBO_INLINE uint32_t randomU32(void)
    {
        return m_uniformUint32(m_randomGen);
    }

    //! Returns a random point in the radius=1 disk centered at the origin of the XY plane.
    float3 CDSF3_VECTORCALL randomInUnitDisk(void)
    {
         // TODO(cort): replace with deterministic algorithm, like octohedron mapping
        float3 p;
        do
        {
            p = float3(
                random01() * 2.0f - 1.0f,
                random01() * 2.0f - 1.0f,
                0);
         } while(lengthSq(p) >= 1.0f);
        return p;
    }

    //! Returns a random point in the radius=1 sphere centered at the origin.
    float3 CDSF3_VECTORCALL randomInUnitSphere(void)
    {
         // TODO(cort): replace with deterministic algorithm, like octohedron mapping
        float3 p;
        do
        {
            p = float3(
                random01() * 2.0f - 1.0f,
                random01() * 2.0f - 1.0f,
                random01() * 2.0f - 1.0f);
        } while(lengthSq(p) >= 1.0f);
        return p;
    }

private:
    std::default_random_engine m_randomGen;
    std::uniform_real_distribution<float> m_uniform;
    std::uniform_int_distribution<uint32_t> m_uniformUint32;
};

static CDS_THREADLOCAL RNG *tls_rng = nullptr;

struct Ray
{
    Ray() {}
    explicit Ray(float3 origin, float3 dir, float time) : origin(origin), dir(dir), invDir(1.0f / dir), time(time) {}
    float3 eval(float t) const { return origin + t*dir; }

    float3 origin, dir, invDir;
    float time;
};

class Camera
{
public:
    struct Params
    {
        float3 eyePos;
        float3 target;
        float3 up;
        float fovDegreesV;
        float aspectRatio;
        float apertureDiameter;
        float focusDistance;
        float exposureSeconds; ///! in seconds
    };
    explicit Camera(const Params &init)
    {
        lensRadius = init.apertureDiameter * 0.5f;
        exposureSeconds = init.exposureSeconds;
        float theta = float(init.fovDegreesV * M_PI/180);
        float halfHeight = tanf(theta/2);
        float halfWidth = init.aspectRatio * halfHeight;
        pos = init.eyePos;
        float3 unitBack  = normalize(init.eyePos - init.target);
        unitRight = normalize(cross(init.up, unitBack));
        unitUp = cross(unitBack,unitRight);
        lowerLeft = init.eyePos - init.focusDistance*(halfWidth*unitRight + halfHeight*unitUp + unitBack);
        horizontal = 2 * halfWidth * init.focusDistance * unitRight;
        vertical = 2 * halfHeight * init.focusDistance * unitUp;
    }
    Ray CDSF3_VECTORCALL rayTo(float u01, float v01, float atTime) const
    {
        float3 rd = lensRadius * tls_rng->randomInUnitDisk();
        float3 offset = unitRight * rd.x() + unitUp * rd.y();
        return Ray(pos+offset, lowerLeft + u01*horizontal + v01*vertical - pos - offset, atTime + tls_rng->random01() * exposureSeconds);
    }
    float3 pos;
    float3 lowerLeft;
    float3 horizontal;
    float3 vertical;
    float3 unitRight;
    float3 unitUp;
    float lensRadius;
    float exposureSeconds;
};

class Material;
struct HitRecord
{
    float t;
    float3 pos;
    float3 normal;
    const Material *pMaterial;
};

struct AABB
{
    AABB() {}
    AABB(float3 minCorner, float3 maxCorner) : minCorner(minCorner), maxCorner(maxCorner)
    {
        assert(minCorner.x() <= maxCorner.x());
        assert(minCorner.y() <= maxCorner.y());
        assert(minCorner.z() <= maxCorner.z());
    }

    float3 minCorner, maxCorner;
};

bool CDSF3_VECTORCALL intersectRayBox(Ray ray, AABB aabb, float tMin, float tMax)
{
#if 1
    float3 d0 = (aabb.minCorner - ray.origin) * ray.invDir;
    float3 d1 = (aabb.maxCorner - ray.origin) * ray.invDir;
    
    float3 v0 = vmin(d0, d1);
    float3 v1 = vmax(d0, d1);

    float dmin = hmax(v0);
    float dmax = hmin(v1);

    bool hit = (dmax >= tMin) && (dmax >= dmin) && (dmin <= tMax);
    return hit;
#else
    for(int i=0; i<3; ++i)
    {
        float t0 = (aabb.minCorner[i] - ray.origin[i]) * ray.invDir[i];
        float t1 = (aabb.maxCorner[i] - ray.origin[i]) * ray.invDir[i];
        if (ray.invDir[i] < 0.0f)
            std::swap(t0,t1);
        tMin = max(tMin, t0);
        tMax = min(tMax, t1);
        if (tMax <= tMin)
            return false;
    }
    return true;
#endif
}

class Hittee
{
public:
    virtual bool boundingBox(float tMin, float tMax, AABB *outAabb) const = 0;
    virtual bool CDSF3_VECTORCALL hit(const Ray ray, float tMin, float tMax, HitRecord *outRecord) const = 0;
};

struct HitteeAndAabb
{
    AABB aabb;
    Hittee *hittee;
};
class BvhNode : public Hittee
{
public:
    BvhNode() = delete;
    BvhNode(HitteeAndAabb hittees[], int count, int splitAxis = 0)
    {
        if (count == 0)
        {
            m_left = m_right = nullptr;
            m_hittee = nullptr;
            return;
        }
        if (count == 1)
        {
            m_hittee = hittees[0].hittee;
            m_left = m_right = nullptr;
            m_aabb = hittees[0].aabb;
            return;
        }
        if      (splitAxis == 0)
            qsort(hittees, count, sizeof(HitteeAndAabb), compareHitteeAabbX);
        else if (splitAxis == 1)
            qsort(hittees, count, sizeof(HitteeAndAabb), compareHitteeAabbY);
        else if (splitAxis == 2)
            qsort(hittees, count, sizeof(HitteeAndAabb), compareHitteeAabbZ);
        m_hittee = nullptr;
        m_left  = new BvhNode(hittees + 0, count/2, (splitAxis+1) % 3);
        m_right = new BvhNode(hittees + count/2, count - count/2, (splitAxis+1) % 3);
        m_aabb = AABB(
            vmin(m_left->m_aabb.minCorner, m_right->m_aabb.minCorner),
            vmax(m_left->m_aabb.maxCorner, m_right->m_aabb.maxCorner));
    }
    virtual ~BvhNode()
    {
        // m_hittee is not owned by the node
        delete m_left;
        delete m_right;
    }
    virtual bool boundingBox(float tMin, float tMax, AABB *outAabb) const
    {
        (void)tMin;
        (void)tMax;
        if (outAabb)
            *outAabb = m_aabb;
        return true;
    }
    virtual bool CDSF3_VECTORCALL hit(const Ray ray, float tMin, float tMax, HitRecord *outRecord) const
    {
        if (m_hittee)
        {
            return m_hittee->hit(ray, tMin, tMax, outRecord);
        }
        if (intersectRayBox(ray, m_aabb, tMin, tMax))
        {
            HitRecord hitRecL, hitRecR;
            bool hitL = m_left  ?  m_left->hit(ray, tMin, tMax, &hitRecL) : false;
            bool hitR = m_right ? m_right->hit(ray, tMin, tMax, &hitRecR) : false;
            if (hitL && hitR)
            {
                *outRecord = (hitRecL.t < hitRecR.t) ? hitRecL : hitRecR;
                return true;
            }
            else if (hitL)
            {
                *outRecord = hitRecL;
                return true;
            }
            else if (hitR)
            {
                *outRecord = hitRecR;
                return true;
            }
        }
        return false;
    }

private:
    Hittee *m_hittee; // non-NULL for leaf nodes only
    BvhNode *m_left, *m_right; // both NULL for leaf nodes. Both non-NULL for internal nodes
    AABB m_aabb;

    static int compareHitteeAabbX(const void *a, const void *b)
    {
        const HitteeAndAabb *ha = (const HitteeAndAabb*)a;
        const HitteeAndAabb *hb = (const HitteeAndAabb*)b;
        float avgA = (ha->aabb.minCorner.x() + ha->aabb.maxCorner.x()) * 0.5f;
        float avgB = (hb->aabb.minCorner.x() + hb->aabb.maxCorner.x()) * 0.5f;
        if (avgA < avgB) return -1;
        if (avgA > avgB) return 1;
        return 0;
    }
    static int compareHitteeAabbY(const void *a, const void *b)
    {
        const HitteeAndAabb *ha = (const HitteeAndAabb*)a;
        const HitteeAndAabb *hb = (const HitteeAndAabb*)b;
        float avgA = (ha->aabb.minCorner.y() + ha->aabb.maxCorner.y()) * 0.5f;
        float avgB = (hb->aabb.minCorner.y() + hb->aabb.maxCorner.y()) * 0.5f;
        if (avgA < avgB) return -1;
        if (avgA > avgB) return 1;
        return 0;
    }
    static int compareHitteeAabbZ(const void *a, const void *b)
    {
        const HitteeAndAabb *ha = (const HitteeAndAabb*)a;
        const HitteeAndAabb *hb = (const HitteeAndAabb*)b;
        float avgA = (ha->aabb.minCorner.z() + ha->aabb.maxCorner.z()) * 0.5f;
        float avgB = (hb->aabb.minCorner.z() + hb->aabb.maxCorner.z()) * 0.5f;
        if (avgA < avgB) return -1;
        if (avgA > avgB) return 1;
        return 0;
    }
};

class Scene : public Hittee
{
public:
    Scene() = delete;
    explicit Scene(const std::vector<Hittee*> &hittees)
        :   m_bvhRoot(nullptr)
    {
        m_withAabb.reserve(hittees.size());
        m_withoutAabb.reserve(hittees.size());
        for(auto itor = hittees.cbegin(); itor != hittees.cend(); itor++)
        {
#if 1 // 0 = disable BVH; put all Hittees in the flat unordered list, to test BVH performance/correctness
            if ((*itor)->boundingBox(0,0,nullptr))
                m_withAabb.push_back(*itor);
            else
#endif
                m_withoutAabb.push_back(*itor);
        }
        std::vector<Hittee*>(m_withAabb).swap(m_withAabb);
        std::vector<Hittee*>(m_withoutAabb).swap(m_withoutAabb);
    }
    void updateBvh(float tMin, float tMax)
    {
        if (m_bvhRoot)
        {
            delete m_bvhRoot;
            m_bvhRoot = nullptr;
        }
        if (!m_withAabb.empty())
        {
            HitteeAndAabb *haa = new HitteeAndAabb[m_withAabb.size()];
            for(int i=0; i<m_withAabb.size(); ++i)
            {
                haa[i].hittee = m_withAabb[i];
                bool hasAabb = m_withAabb[i]->boundingBox(tMin, tMax, &haa[i].aabb);
                (void)hasAabb;
                assert(hasAabb);
            }
            m_bvhRoot = new BvhNode(haa, (int)m_withAabb.size());
            delete [] haa;
        }
    }
    virtual bool boundingBox(float tMin, float tMax, AABB *outAabb) const
    {
        (void)tMin;
        (void)tMax;
        (void)outAabb;
        return false;
    }
    virtual bool CDSF3_VECTORCALL hit(const Ray ray, float tMin, float tMax, HitRecord *outRecord) const
    {
        bool hitSomething = false;
        float closestSoFar = tMax;
        if (m_bvhRoot)
            hitSomething = m_bvhRoot->hit(ray, tMin, tMax, outRecord);
        if (!outRecord && hitSomething)
            return true;
        for(auto itor = m_withoutAabb.begin(); itor != m_withoutAabb.end(); ++itor)
        {
            if ((*itor)->hit(ray, tMin, closestSoFar, outRecord))
            {
                hitSomething = true;
                if (!outRecord)
                    return true;
                closestSoFar = outRecord->t;
            }
        }
        return hitSomething;
    }

private:
    BvhNode *m_bvhRoot;
    std::vector<Hittee*> m_withAabb;
    std::vector<Hittee*> m_withoutAabb;
};

class Material
{
public:
    //! Determines whether a ray should be scattered from the specified input ray and hit record. If so, output the scattered ray and attenuation factor.
    virtual bool CDSF3_VECTORCALL scatter(const Ray rayIn, const HitRecord &hit, float3 *outAttenuation, Ray *outRay) const = 0;
};

class LambertianMaterial : public Material
{
public:
    explicit LambertianMaterial(float3 albedo) : albedo(albedo) {}
    bool CDSF3_VECTORCALL scatter(const Ray rayIn, const HitRecord &hit, float3 *outAttenuation, Ray *outRay) const override
    {
        // scatter towards a random point in the unit sphere above the hit point
        float3 target = hit.pos + hit.normal + tls_rng->randomInUnitSphere();
        *outRay = Ray(hit.pos, target-hit.pos, rayIn.time);
        *outAttenuation = albedo;
        return true;
    }

    float3 albedo;
};

class MetalMaterial : public Material
{
public:
    explicit MetalMaterial(float3 albedo, float roughness) : albedo(albedo), roughness(clamp(roughness,0.0f,1.0f)) {}
    bool CDSF3_VECTORCALL scatter(const Ray rayIn, const HitRecord &hit, float3 *outAttenuation, Ray *outRay) const override
    {
        float3 reflectDir = reflect(normalize(rayIn.dir), hit.normal);
        *outRay = Ray(hit.pos, reflectDir + roughness*tls_rng->randomInUnitSphere(), rayIn.time);
        *outAttenuation = albedo;
        return dot(outRay->dir, hit.normal) > 0;
    }
    float3 albedo;
    float roughness;
};

class DieletricMaterial : public Material
{
public:
    explicit DieletricMaterial(float3 albedo, float refractionIndex) : albedo(albedo), refractionIndex(refractionIndex) {}
    bool CDSF3_VECTORCALL scatter(const Ray rayIn, const HitRecord &hit, float3 *outAttenuation, Ray *outRay) const override
    {
        float3 nOut;
        float niOverNt;
        float cosine;
        float rayDotNormal = dot(rayIn.dir, hit.normal);
        // TODO(cort): doesn't correctly handle transitions between dielectric <-> non-air materials
        if (rayDotNormal > 0) // exiting object into air
        {
            nOut = -hit.normal;
            niOverNt = refractionIndex;
            cosine = rayDotNormal / length(rayIn.dir);
            cosine = sqrtf(1.0f - refractionIndex*refractionIndex*(1.0f-cosine*cosine));
        }
        else // entering object from air
        {
            nOut = hit.normal;
            niOverNt = 1.0f / refractionIndex;
            cosine = -rayDotNormal / length(rayIn.dir);
        }

        float3 refractedDir;
        float reflectionChance;
        *outAttenuation = albedo; // dielectrics absorb nothing. TODO(cort) but what about colored glass?
        if (refract(rayIn.dir, nOut, niOverNt, &refractedDir))
        {
            reflectionChance = schlick(cosine);
        }
        else
        {
            reflectionChance = 1.0f;
        }
        if (tls_rng->random01() < reflectionChance)
        {
            *outRay = Ray(hit.pos, reflect(rayIn.dir, hit.normal), rayIn.time);
        }
        else
        {
            *outRay = Ray(hit.pos, refractedDir, rayIn.time);
        }
        return true;
    }
    float3 albedo;
    float refractionIndex;
private:
    inline float schlick(float cosine) const
    {
        float r0 = (1-refractionIndex) / (1+refractionIndex);
        r0 *= r0;
        return r0 + (1-r0)*pow((1-cosine), 5);
    }
};

class Sphere : public Hittee
{
public:
    Sphere() = delete;
    explicit Sphere(const AnimationChannel<float3> &center, float radius, const Material *material) : center(center), radius(radius), material(material) {}
    virtual bool boundingBox(float tMin, float tMax, AABB *outAabb) const
    {
        if (outAabb)
        {
            float absRadius = fabsf(radius);
            float3 radius3 = float3(absRadius, absRadius, absRadius);
            float3 centerMin = center.eval(tMin);
            float3 centerMax = center.eval(tMax);
            float3 bbMin = vmin(centerMin-radius3, centerMax-radius3);
            float3 bbMax = vmax(centerMin+radius3, centerMax+radius3);
            // TODO(cort): This assumes motion between tMin and tMaxx is a straight line.
            // It would be safer to sample the position at intermediate points in the range [tMin,tMax] and expand the
            // bounding box accordingly. Or if there's an analytical solution, even better!
            *outAabb = AABB(bbMin, bbMax);
        }
        return true;
    }
    virtual bool CDSF3_VECTORCALL hit(const Ray ray, float tMin, float tMax, HitRecord *outRecord) const
    {
        float3 centerNow = center.eval(ray.time);
        float3 oc = ray.origin - centerNow;
        float a = dot(ray.dir, ray.dir);
        float b = dot(oc, ray.dir);
        float c = dot(oc, oc) - radius*radius;
        float discriminant = b*b - a*c;
        if (discriminant > 0)
        {
            float temp = sqrtf(discriminant);
            float t0 = (-b - temp) / a;
            if (tMin <= t0 && t0 <= tMax)
            {
                if (outRecord)
                {
                    outRecord->t = t0;
                    outRecord->pos = ray.eval(t0);
                    outRecord->normal = (outRecord->pos - centerNow) / radius;
                    outRecord->pMaterial = material;
                }
                return true;
            }
            float t1 = (-b + temp) / a;
            if (tMin <= t1 && t1 <= tMax)
            {
                if (outRecord)
                {
                    outRecord->t = t1;
                    outRecord->pos = ray.eval(t1);
                    outRecord->normal = (outRecord->pos - centerNow) / radius;
                    outRecord->pMaterial = material;
                }
                return true;
            }
        }
        return false;
    }

    AnimationChannel<float3> center;
    float radius;
    const Material *material;
};

float3 CDSF3_VECTORCALL rayColor(const Ray ray, const Hittee *scene, int depth = 0)
{
    const int kMaxScatterDepth = 50;
    HitRecord hit;
    if (scene->hit(ray, 0.001f, FLT_MAX, &hit)) // TODO(cort): proper epsilon
    {
        Ray scatterRay;
        float3 scatterAttenuation;
        if (depth < kMaxScatterDepth && hit.pMaterial->scatter(ray, hit, &scatterAttenuation, &scatterRay))
        {
            return scatterAttenuation * rayColor(scatterRay, scene, depth+1);
        }
        else
        {
            return float3(0,0,0);
        }
    }
    else
    {
        // background color
        float3 unitDir = normalize(ray.dir);
        float t = 0.5f * (unitDir.y() + 1.0f);
        return lerp( float3(1,1,1), float3(0.5f, 0.7f, 1.0f), t);
    }


}

static const char *filenameSuffix(const char *filename)
{
    const char *suffix = filename + strlen(filename) - 1;
    while(suffix != filename)
    {
        if (*suffix == '.')
            return suffix+1;
        --suffix;
    }
    return NULL;
}

struct RenderSettings
{
    const Camera *camera;
    const Hittee *scene;
    float *imgPixels;
    float time;
    int imgWidth;
    int imgHeight;
    int samplesPerPixel;
};
struct JobArgs
{
    unsigned long randomSeed;
    int x0, x1, y0, y1;
};

struct WorkerArgs
{
    RenderSettings *renderSettings;
    const JobArgs *jobs;
    std::atomic<int> *nextJobIndex;
    int jobCount;
    unsigned int randomSeed;
};
void workerFunc(WorkerArgs *threadArgs)
{
    tls_rng = new RNG(threadArgs->randomSeed);

    const RenderSettings &render = *(threadArgs->renderSettings);
    const float kPixelOffsetScaleX =  2.0f / (float)render.imgWidth;
    const float kPixelOffsetScaleY =  2.0f / (float)render.imgHeight;
    const float kPixelOffsetBiasX  = -1.0f / (float)render.imgWidth;
    const float kPixelOffsetBiasY  = -1.0f / (float)render.imgHeight;
    auto startTime = std::chrono::high_resolution_clock::now();
    int threadJobCount = 0;
    std::atomic<int> &nextJobIndex = *(threadArgs->nextJobIndex);
    for(;;)
    {
        int jobIndex = nextJobIndex++;
        if (jobIndex >= threadArgs->jobCount)
            break;
        threadJobCount += 1;
        const JobArgs &job = threadArgs->jobs[jobIndex];
        tls_rng->seed(job.randomSeed);
        for(int iY=job.y0; iY<job.y1; iY+=1)
        {
            for(int iX=job.x0; iX<job.x1; iX+=1)
            {
                float u = float(iX) / float(render.imgWidth-1);
                float v = 1.0f - float(iY) / float(render.imgHeight-1);
                float3 color(0,0,0);
                for(int iS=0; iS<render.samplesPerPixel; ++iS)
                {
                    Ray ray = render.camera->rayTo(
                        u + (tls_rng->random01() * kPixelOffsetScaleX + kPixelOffsetBiasX),
                        v + (tls_rng->random01() * kPixelOffsetScaleY + kPixelOffsetBiasY),
                        render.time);
                    color += rayColor(ray, render.scene);
                }
                color /= (float)render.samplesPerPixel;

                float *out = render.imgPixels + 4*(render.imgWidth*iY + iX);
                color.store( out );
                out[3] = 1.0f;
            }
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime-startTime).count();
    printf("Thread finished %3d jobs in %.3f seconds (%.3f ms per job)\n", threadJobCount, double(elapsedNanos)/1e9,
        double(elapsedNanos)/ (1e6 * double(threadJobCount)));

    delete tls_rng;
}

static void usage(const char *argv0)
{
    printf("usage: %s [ARGS]\n", argv0);
    printf("-seed N   Seed RNG with N [default: current time]\n");
    printf("-out F    Specify output file [default: out.png]\n");
    printf("          Output file format is determined from the file suffix.\n");
    printf("          Supported formats: PNG, BMP, TGA, HDR\n");
    printf("-preview  Render a low-quality preview, to quickly verify scene composition.\n");
}

int main(int argc, char *argv[])
{
    int kOutputWidth  = 800;
    int kOutputHeight = 600;
#ifdef _DEBUG
    int kSamplesPerPixel = 10;
#else
    int kSamplesPerPixel = 100;
#endif

    const char *outputFilename = "out.png";
    unsigned int randomSeed = (unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for(int iArg=1; iArg<argc; ++iArg)
    {
        if (0 == strncmp(argv[iArg], "--help", 7) ||
            0 == strncmp(argv[iArg], "-h", 3))
        {
            usage(argv[0]);
            return 0;
        }
        else if (0 == strncmp(argv[iArg], "-seed", 6) && iArg+1 < argc)
        {
            randomSeed = (unsigned long)strtol(argv[++iArg], NULL, 10);
            continue;
        }
        else if (0 == strncmp(argv[iArg], "-out", 5) && iArg+1 < argc)
        {
            outputFilename = argv[++iArg];
            continue;
        }
        else if (0 == strncmp(argv[iArg], "-preview", 9))
        {
            kSamplesPerPixel = 1;
        }
    }
    const char *outputFilenameSuffix = filenameSuffix(outputFilename);

    tls_rng = new RNG(randomSeed);

    Camera::Params cameraParams = {};
    cameraParams.eyePos      = float3( 0, 2, 5);
    cameraParams.target      = float3( 0, 0.5f, 0);
    cameraParams.up          = float3( 0, 1, 0);
    cameraParams.fovDegreesV = 45.0f;
    cameraParams.aspectRatio = (float)kOutputWidth / (float)kOutputHeight;
    cameraParams.apertureDiameter = 0.03f;
    cameraParams.focusDistance = length(cameraParams.target-cameraParams.eyePos);
    cameraParams.exposureSeconds = 1.0f / 30.0f;
    Camera camera(cameraParams);
    const float captureTime = 0.5f; // what time should the Camera's virtual shutter open?

    std::vector<Hittee*> contents;
    {
        LambertianMaterial greenLambert(float3(0.3f, 0.8f, 0.3f));

        std::vector<Material*> randomMaterials;
        const int randomMaterialCount = 100;
        for(int iMat=0; iMat<randomMaterialCount; ++iMat)
        {
            switch(tls_rng->randomU32() % 3)
            {
            case 0:
                randomMaterials.push_back(new LambertianMaterial(
                    float3(tls_rng->random01(), tls_rng->random01(), tls_rng->random01()) // albedo
                    ));
                break;
            case 1:
                randomMaterials.push_back(new MetalMaterial(
                    float3(tls_rng->random01(), tls_rng->random01(), tls_rng->random01()), // albedo
                    powf(tls_rng->random01(), 4.0f) // roughness (higher exponent -> bias towards lower roughness)
                    ));
                break;
            case 2:
                randomMaterials.push_back(new DieletricMaterial(
                    float3(tls_rng->random01(), tls_rng->random01(), tls_rng->random01()), // albedo
                    1.0f + powf(tls_rng->random01(), 7.0f) // refraction index. Currently clamped to [1..2], biased towards 1.0.
                    ));
                break;
            }
        }

        contents.push_back(new Sphere(
            float3(0.0f, -100, 00.0f), // center
            100.0f, // radius
            new LambertianMaterial(float3(0.3f, 0.8f, 0.3f)) // material
            )); // ground sphere

        const int randomSceneObjectCountX = 30;
        const int randomSceneObjectCountZ = 30;
        const float randomSceneObjectSpacingX = 1.0f;
        const float randomSceneObjectSpacingZ = 1.0f;

        const float randomSceneObjectMaxRadius = min(randomSceneObjectSpacingX, randomSceneObjectSpacingZ) * 0.3f;
        const float randomSceneObjectMaxShiftXZ = 1.0f - 2.0f*randomSceneObjectMaxRadius;
        const float3 randomPosBias(
            float(-randomSceneObjectCountX)*0.5f*randomSceneObjectSpacingX,
            0,
            float(-randomSceneObjectCountZ)*0.5f*randomSceneObjectSpacingZ);
        for(int iZ=0; iZ<randomSceneObjectCountZ; iZ+=1)
        {
            for(int iX=0; iX<randomSceneObjectCountZ; iX+=1)
            {
                float radius = tls_rng->random01() * randomSceneObjectMaxRadius;
                float xShift = (tls_rng->random01() - 0.5f) * 2.0f * randomSceneObjectMaxShiftXZ;
                float yShift = tls_rng->random01() * cameraParams.eyePos.y() * 0.5f;
                float zShift = (tls_rng->random01() - 0.5f) * 2.0f * randomSceneObjectMaxShiftXZ;
                float3 basePos = randomPosBias + float3(
                    float(iX)*randomSceneObjectSpacingX + xShift,
                    radius + yShift,
                    float(iZ)*randomSceneObjectSpacingZ + zShift);
                float bounceHeight = tls_rng->random01() < 0.2f ? powf(tls_rng->random01(), 3.0f) : 0.0f; // set to 0 to disable bounce / motion blur
                contents.push_back(new Sphere(
                    AnimationChannel<float3>({
                        {0.0f, basePos+float3(0,0,0)},
                        {1.0f, basePos+float3(0,bounceHeight,0)},
                    }),
                    radius,
                    randomMaterials[ tls_rng->randomU32() % randomMaterialCount ]
                    ));
            }
        }
    }

    Scene scene(contents);
    scene.updateBvh(captureTime, captureTime+camera.exposureSeconds);

    float *outputPixels = new float[kOutputWidth * kOutputHeight * 4];

    const int kThreadCount = zomboCpuCount()*1;
    std::vector<std::thread> threads(kThreadCount);
    std::vector<WorkerArgs> threadArgs(kThreadCount);

    auto startTicks = std::chrono::high_resolution_clock::now();

    const int totalJobCount = ((kOutputWidth+31)/32) * ((kOutputHeight+31)/32);
    JobArgs *jobs = new JobArgs[totalJobCount];
    int iJob=0;
    for(int iY=0; iY<kOutputHeight; iY+=32)
    {
        for(int iX=0; iX<kOutputWidth; iX+=32)
        {
            jobs[iJob] = JobArgs();
            jobs[iJob].randomSeed = tls_rng->randomU32();
            jobs[iJob].x0 = iX;
            jobs[iJob].y0 = iY;
            jobs[iJob].x1 = min(iX+32, kOutputWidth);
            jobs[iJob].y1 = min(iY+32, kOutputHeight);
            iJob += 1;
        }
    }
    std::atomic<int> nextJobIndex(0);

    RenderSettings render = RenderSettings();
    render.camera = &camera;
    render.scene = &scene;
    render.imgPixels = outputPixels;
    render.imgWidth = kOutputWidth;
    render.imgHeight = kOutputHeight;
    render.time = captureTime;
    render.samplesPerPixel = kSamplesPerPixel;
    for(int iThread=0; iThread<kThreadCount; ++iThread)
    {
        threadArgs[iThread] = WorkerArgs();
        threadArgs[iThread].renderSettings = &render;
        threadArgs[iThread].jobs = jobs;
        threadArgs[iThread].nextJobIndex = &nextJobIndex;
        threadArgs[iThread].jobCount = totalJobCount;
        threadArgs[iThread].randomSeed = tls_rng->randomU32();

        threads[iThread] = std::thread(workerFunc, &threadArgs[iThread]);
    }
    for(int iThread=0; iThread<kThreadCount; ++iThread)
    {
        threads[iThread].join();
    }
    delete [] jobs;
    auto endTicks = std::chrono::high_resolution_clock::now();

    int imageWriteSuccess = 0;
    if (zomboStrncasecmp(outputFilenameSuffix, "hdr", 3) == 0)
    {
        imageWriteSuccess = stbi_write_hdr(outputFilename, kOutputWidth, kOutputHeight, 4, outputPixels);
    }
    else
    {
        uint32_t *normalizedPixels = new uint32_t[kOutputWidth*kOutputHeight];
        for(int i=0; i<kOutputWidth*kOutputHeight; ++i)
        {
            float3 color = float3(outputPixels + 4*i);
            color = clamp(color, float3(0,0,0), float3(1,1,1));
            color = float3( sqrtf(color.x()), sqrtf(color.y()), sqrtf(color.z()) ); // gamma-"correct" with gamma=2.0
            normalizedPixels[i] =
                ( uint8_t(255.99f*color.r()) <<  0 ) |
                ( uint8_t(255.99f*color.g()) <<  8 ) |
                ( uint8_t(255.99f*color.b()) << 16 ) |
                ( 0xFF                       << 24 );
        }
        if (zomboStrncasecmp(outputFilenameSuffix, "bmp", 3) == 0)
            imageWriteSuccess = stbi_write_bmp(outputFilename, kOutputWidth, kOutputHeight, 4, normalizedPixels);
        if (zomboStrncasecmp(outputFilenameSuffix, "png", 3) == 0)
            imageWriteSuccess = stbi_write_png(outputFilename, kOutputWidth, kOutputHeight, 4, normalizedPixels, kOutputWidth*sizeof(uint32_t));
        if (zomboStrncasecmp(outputFilenameSuffix, "tga", 3) == 0)
            imageWriteSuccess = stbi_write_tga(outputFilename, kOutputWidth, kOutputHeight, 4, normalizedPixels);
        delete [] normalizedPixels;
    }
    assert(imageWriteSuccess);
    delete [] outputPixels;

    auto elapsedNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(endTicks-startTicks).count();
    printf("Rendered %s [%dx%d %ds/p] in %.3f seconds\n", outputFilename,
        kOutputWidth, kOutputHeight, kSamplesPerPixel, double(elapsedNanos)/1e9);
    return 0;
}
