#include "cds_float3.h"
#include "platform.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _MSC_VER
#   include <windows.h>
#else

#endif

#include <assert.h>
#include <chrono>
#include <float.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <vector>

#if !defined(M_PI)
#   define M_PI 3.14159265358979323846
#endif

static inline float clamp(float x, float a, float b)
{
    return (x<a) ? a : (x>b ? b : x);
}

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
        ,   m_biuniform(-1.0f, 1.0f)
        ,   m_uniformUint32(0, UINT_MAX)
    {}

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
            p = float3( m_biuniform(m_randomGen), m_biuniform(m_randomGen), 0 );
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
            p = float3( m_biuniform(m_randomGen), m_biuniform(m_randomGen), m_biuniform(m_randomGen) );
        } while(lengthSq(p) >= 1.0f);
        return p;
    }

private:
    std::default_random_engine m_randomGen;
    std::uniform_real_distribution<float> m_uniform;
    std::uniform_real_distribution<float> m_biuniform;
    std::uniform_int_distribution<uint32_t> m_uniformUint32;
};
static RNG *g_rng = nullptr;

struct Ray
{
    Ray() {}
    explicit Ray(float3 origin, float3 dir) : origin(origin), dir(dir) {}
    float3 eval(float t) const { return origin + t*dir; }

    float3 origin, dir;
};

class Camera
{
public:
    explicit Camera(float3 eyePos, float3 target, float3 up, float fovV, float aspectRatio, float aperture, float focusDistance)
    {
        lensRadius = aperture/2;
        float theta = float(fovV * M_PI/180);
        float halfHeight = tanf(theta/2);
        float halfWidth = aspectRatio * halfHeight;
        pos = eyePos;
        float3 unitBack  = normalize(eyePos - target);
        unitRight = normalize(cross(up, unitBack));
        unitUp = cross(unitBack,unitRight);
        lowerLeft = eyePos - focusDistance*(halfWidth*unitRight + halfHeight*unitUp + unitBack);
        horizontal = 2 * halfWidth * focusDistance * unitRight;
        vertical = 2 * halfHeight * focusDistance * unitUp;
    }
    Ray CDSF3_VECTORCALL ray(float u, float v) const
    {
        float3 rd = lensRadius * g_rng->randomInUnitDisk();
        float3 offset = unitRight * rd.x() + unitUp * rd.y();
        return Ray(pos+offset, lowerLeft + u*horizontal + v*vertical - pos - offset);
    }
    float3 pos;
    float3 lowerLeft;
    float3 horizontal;
    float3 vertical;
    float3 unitRight;
    float3 unitUp;
    float lensRadius;
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
    bool CDSF3_VECTORCALL scatter(const Ray /*rayIn*/, const HitRecord &hit, float3 *outAttenuation, Ray *outRay) const override
    {
        // scatter towards a random point in the unit sphere above the hit point
        float3 target = hit.pos + hit.normal + g_rng->randomInUnitSphere();
        *outRay = Ray(hit.pos, target-hit.pos);
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
        *outRay = Ray(hit.pos, reflectDir + roughness*g_rng->randomInUnitSphere());
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
        if (g_rng->random01() < reflectionChance)
        {
            *outRay = Ray(hit.pos, reflect(rayIn.dir, hit.normal));
        }
        else
        {
            *outRay = Ray(hit.pos, refractedDir);
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
    Sphere() {}
    explicit Sphere(float3 center, float radius, const Material *material) : center(center), radius(radius), material(material) {}
    virtual bool CDSF3_VECTORCALL hit(const Ray ray, float tMin, float tMax, HitRecord *outRecord) const
    {
        float3 oc = ray.origin - center;
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
                outRecord->normal = (outRecord->pos - center) / radius;
                outRecord->pMaterial = material;
                return true;
            }
            float t1 = (-b + temp) / a;
            if (tMin <= t1 && t1 <= tMax)
            {
                outRecord->t = t1;
                outRecord->pos = ray.eval(t1);
                outRecord->normal = (outRecord->pos - center) / radius;
                outRecord->pMaterial = material;
                return true;
            }
        }
        return false;
    }

    float3 center;
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

static void usage(const char *argv0)
{
    printf("usage: %s [ARGS]\n", argv0);
    printf("-seed N   Seed RNG with N [default: current time]\n");
    printf("-out F    Specify output file [default: out.png]\n");
    printf("          Output file format is determined from the file suffix.\n");
    printf("          Supported formats: PNG, BMP, TGA, HDR\n");
}

int main(int argc, char *argv[])
{
#if 0
    // float3 test code
    float hitT = 10000000.0f;
    bool hit = intersectRayBox(float3(0,0,0), float3(1, 1, 1) / float3(1,0,0), float3(-1,-1,-1), float3(1,1,1), hitT);
    printf("hit %i at t=%f\n", hit, hitT);
    return 0;
#endif

    const char *outputFilename = "out.png";
    unsigned int randomSeed = (unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for(int iArg=0; iArg<argc; ++iArg)
    {
        if (0 == strncmp(argv[iArg], "--help", 6) ||
            0 == strncmp(argv[iArg], "-h", 2))
        {
            usage(argv[0]);
            return 0;
        }
        else if (0 == strncmp(argv[iArg], "-seed", 5) && iArg+1 < argc)
        {
            randomSeed = (unsigned long)strtol(argv[++iArg], NULL, 10);
            continue;
        }
        else if (0 == strncmp(argv[iArg], "-out", 4) && iArg+1 < argc)
        {
            outputFilename = argv[++iArg];
            continue;
        }
    }
    const char *outputFilenameSuffix = filenameSuffix(outputFilename);

    const int kOutputWidth  = 800;
    const int kOutputHeight = 600;
#ifdef _DEBUG
    const int kSamplesPerPixel = 10;
#else
    const int kSamplesPerPixel = 100;
#endif

    g_rng = new RNG(randomSeed);

    std::default_random_engine randomGen;
    randomGen.seed(randomSeed);
    std::uniform_real_distribution<float> randomPixelOffsetX(-1.0f/(float)kOutputWidth,  1.0f/(float)kOutputWidth);
    std::uniform_real_distribution<float> randomPixelOffsetY(-1.0f/(float)kOutputHeight, 1.0f/(float)kOutputHeight);

    LambertianMaterial yellowLambert(float3(1,1,0.0));
    LambertianMaterial greenLambert(float3(0.3f, 0.8f, 0.3f));
    MetalMaterial copperMetal(float3(0.8549f, 0.5412f, 0.4039f), 0.4f);
    MetalMaterial silverMetal(float3(0.9f,0.9f,0.9f), 0.05f);
    DieletricMaterial whiteGlass(float3(1,1,1), 1.5f);
    DieletricMaterial yellowGlass(float3(1,1,0.9f), 1.5f);
    auto contents = std::vector<Hittee*>{
        new Sphere( float3(0.0f, -100, 00.0f), 100.0f, &greenLambert ),
        new Sphere( float3(0.0f, 0.5f, 0.0f), 0.5f, &silverMetal ),
        new Sphere( float3(-1.1f, 0.5f, 0.0f), 0.5f, &copperMetal ),
        new Sphere( float3( 1.1f, 0.5f, 0.0f), 0.5f, &whiteGlass ),
        new Sphere( float3( 1.1f, 0.5f, 0.0f), -0.48f, &whiteGlass ),
    };
    HitteeList hittees(contents);

    const float aspectRatio = (float)kOutputWidth / (float)kOutputHeight;
    const float3 camPos    = float3( 2, 1, 1);
    const float3 camTarget = float3( 0, 0.5f, 0);
    const float3 camUp     = float3( 0, 1, 0);
    const float camAperture = 0.03f;
    const float camFocusDistance = length(camTarget-camPos);
    Camera camera(camPos, camTarget, camUp, 45.0f, aspectRatio, camAperture, camFocusDistance);

    float *outputPixels = new float[kOutputWidth * kOutputHeight * 4];

    auto startTime = std::chrono::high_resolution_clock::now();
    for(int iY=0; iY<kOutputHeight; iY+=1)
    {
        for(int iX=0; iX<kOutputWidth; iX+=1)
        {
            float u = float(iX) / float(kOutputWidth-1);
            float v = 1.0f - float(iY) / float(kOutputHeight-1);
            float3 color(0,0,0);
            for(int iS=0; iS<kSamplesPerPixel; ++iS)
            {
                Ray ray = camera.ray(
                    u + randomPixelOffsetX(randomGen),
                    v + randomPixelOffsetY(randomGen));
                color += rayColor(ray, hittees);
            }
            color /= kSamplesPerPixel;

            float *out = outputPixels + 4*(kOutputWidth*iY + iX);
            color.store( out );
            out[3] = 1.0f;
        }
    };
    auto endTime = std::chrono::high_resolution_clock::now();

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

    auto elapsedNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime-startTime).count();
    printf("Rendered %s [%dx%d %ds/p] in %.3f seconds\n", outputFilename,
        kOutputWidth, kOutputHeight, kSamplesPerPixel, double(elapsedNanos)/1e9);
    return 0;
}
