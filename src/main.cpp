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
    explicit Ray(float3 origin, float3 dir, float time) : origin(origin), dir(dir), time(time) {}
    float3 eval(float t) const { return origin + t*dir; }

    float3 origin, dir;
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

class Hittee
{
public:
    virtual bool CDSF3_VECTORCALL hit(const Ray ray, float tMin, float tMax, HitRecord *outRecord) const = 0;
};

class HitteeList : public Hittee
{
public:
    HitteeList() {}
    explicit HitteeList(std::vector<Hittee*> &hittees) : list(std::move(hittees)) {}
    virtual bool CDSF3_VECTORCALL hit(const Ray ray, float tMin, float tMax, HitRecord *outRecord) const
    {
        HitRecord hitRecord;
        bool hitSomething = false;
        float closestSoFar = tMax;
        for(auto itor = list.begin(); itor != list.end(); ++itor)
        {
            if ((*itor)->hit(ray, tMin, closestSoFar, &hitRecord))
            {
                hitSomething = true;
                closestSoFar = hitRecord.t;
            }
        }
        if (hitSomething)
        {
            *outRecord = hitRecord;
        }
        return hitSomething;
    }

    std::vector<Hittee*> list;
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
                outRecord->t = t0;
                outRecord->pos = ray.eval(t0);
                outRecord->normal = (outRecord->pos - centerNow) / radius;
                outRecord->pMaterial = material;
                return true;
            }
            float t1 = (-b + temp) / a;
            if (tMin <= t1 && t1 <= tMax)
            {
                outRecord->t = t1;
                outRecord->pos = ray.eval(t1);
                outRecord->normal = (outRecord->pos - centerNow) / radius;
                outRecord->pMaterial = material;
                return true;
            }
        }
        return false;
    }

    AnimationChannel<float3> center;
    float radius;
    const Material *material;
};


bool CDSF3_VECTORCALL intersectRayBox(float3 rayOrg, float3 invDir, float3 bbmin, float3 bbmax, float &hitT)
{
    float3 d0 = (bbmin - rayOrg) * invDir;
    float3 d1 = (bbmax - rayOrg) * invDir;

    float3 v0 = vmin(d0, d1);
    float3 v1 = vmax(d0, d1);

    float tmin = hmax(v0);
    float tmax = hmin(v1);

    bool hit = (tmax >= 0) && (tmax >= tmin) && (tmin <= hitT);
    if (hit)
        hitT = tmin;
    return hit;
}

float3 CDSF3_VECTORCALL rayColor(const Ray ray, const HitteeList &world, int depth = 0)
{
    const int kMaxScatterDepth = 50;
    HitRecord hit;
    if (world.hit(ray, 0.001f, FLT_MAX, &hit)) // TODO(cort): proper epsilon
    {
        Ray scatterRay;
        float3 scatterAttenuation;
        if (depth < kMaxScatterDepth && hit.pMaterial->scatter(ray, hit, &scatterAttenuation, &scatterRay))
        {
            return scatterAttenuation * rayColor(scatterRay, world, depth+1);
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
    const HitteeList *scene;
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
                    color += rayColor(ray, *(render.scene));
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
    printf("Thread finished %3d jobs in %.3f seconds\n", threadJobCount, double(elapsedNanos)/1e9);

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

    LambertianMaterial yellowLambert(float3(1,1,0.0));
    LambertianMaterial greenLambert(float3(0.3f, 0.8f, 0.3f));
    MetalMaterial copperMetal(float3(0.8549f, 0.5412f, 0.4039f), 0.4f);
    MetalMaterial silverMetal(float3(0.9f,0.9f,0.9f), 0.05f);
    DieletricMaterial whiteGlass(float3(1,1,1), 1.5f);
    DieletricMaterial yellowGlass(float3(1,1,0.9f), 1.5f);
    auto contents = std::vector<Hittee*>{
        new Sphere( float3(0.0f, -100, 00.0f), 100.0f, &greenLambert ),
        new Sphere( float3(0.0f, 0.5f, 0.0f), 0.5f, &silverMetal ),
        //new Sphere( float3(-1.1f, 0.5f, 0.0f), 0.5f, &copperMetal ),
        new Sphere(
            AnimationChannel<float3>({
                {0.0f, float3(-0.1f, 0.5f, 0.0f)},
                {1.0f, float3(-2.1f, 0.5f, 0.0f)},
            }),
            0.5f,
            &copperMetal),
        new Sphere( float3( 1.1f, 0.5f, 0.0f), 0.5f, &whiteGlass ),
        new Sphere( float3( 1.1f, 0.5f, 0.0f), -0.48f, &whiteGlass ),
    };
    const HitteeList scene(contents);

    Camera::Params cameraParams = {};
    cameraParams.eyePos      = float3( 2, 1, 1);
    cameraParams.target      = float3( 0, 0.5f, 0);
    cameraParams.up          = float3( 0, 1, 0);
    cameraParams.fovDegreesV = 45.0f;
    cameraParams.aspectRatio = (float)kOutputWidth / (float)kOutputHeight;
    cameraParams.apertureDiameter = 0.03f;
    cameraParams.focusDistance = length(cameraParams.target-cameraParams.eyePos);
    cameraParams.exposureSeconds = 1.0f / 30.0f;
    Camera camera(cameraParams);

    float *outputPixels = new float[kOutputWidth * kOutputHeight * 4];

    const int kThreadCount = zomboCpuCount()*1;
    std::vector<std::thread> threads(kThreadCount);
    std::vector<WorkerArgs> threadArgs(kThreadCount);
    const float captureTime = 0.5f; // what time should the Camera's virtual shutter open?

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
