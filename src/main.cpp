#include "cds_float3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _MSC_VER
#	include <windows.h>
#endif

#include <assert.h>
#include <chrono>
#include <float.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <vector>

#if !defined(M_PI)
#	define M_PI 3.14159265358979323846
#endif

static inline float clamp(float x, float a, float b)
{
	return (x<a) ? a : (x>b ? b : x);
}

class RNG
{
public:
	RNG() : m_randomGen( (unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count() ),
		m_uniform(0.0f, 1.0f), m_biuniform(-1.0f, 1.0f) {}
	explicit RNG(unsigned long seed) : m_randomGen(seed), m_uniform(0.0f, 1.0f), m_biuniform(-1.0f, 1.0f) {}

	inline float random01(void)
	{
		return m_uniform(m_randomGen);
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
	std::default_random_engine m_randomGen; // TODO(cort): seed correctness?
	std::uniform_real_distribution<float> m_uniform;
	std::uniform_real_distribution<float> m_biuniform;
};
static RNG g_rng;

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
	explicit Camera(float3 eyePos, float3 target, float3 up, float fovV, float aspectRatio)
	{
		float theta = float(fovV * M_PI/180);
		float halfHeight = tanf(theta*0.5f);
		float halfWidth = aspectRatio * halfHeight;
		pos = eyePos;
		float3 camBack  = normalize(eyePos - target);
		float3 camRight = normalize(cross(up, camBack));
		float3 camUp = cross(camBack,camRight);
		lowerLeft = eyePos - halfWidth*camRight - halfHeight*camUp - camBack;
		horizontal = 2*halfWidth*camRight;
		vertical = 2*halfHeight*camUp;
	}
	Ray ray(float u, float v) const
	{
		return Ray(pos, lowerLeft + u*horizontal + v*vertical - pos);
	}
	float3 pos;
	float3 lowerLeft;
	float3 horizontal;
	float3 vertical;
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
		float closestSoFar = FLT_MAX;
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
		float3 target = hit.pos + hit.normal + g_rng.randomInUnitSphere();
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
		*outRay = Ray(hit.pos, reflectDir + roughness*g_rng.randomInUnitSphere());
		*outAttenuation = albedo;
		return dot(outRay->dir, hit.normal) > 0;
	}
	float3 albedo;
	float roughness;
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

float3 CDSF3_VECTORCALL rayColor(const Ray ray, HitteeList &world, int depth = 0)
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
		return lerp( float3(1,1,1), float3(0.5, 0.7, 1.0), t);
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

int __cdecl main(int argc, char *argv[])
{
#if 0
	// float3 test code
	float hitT = 10000000.0f;
	bool hit = intersectRayBox(float3(0,0,0), float3(1, 1, 1) / float3(1,0,0), float3(-1,-1,-1), float3(1,1,1), hitT);
	printf("hit %i at t=%f\n", hit, hitT);
	return 0;
#endif

	if (argc < 2)
	{
		printf("usage: %1 [output.hdr]\n", argv[0]);
		return 0;
	}
	const char *outputFilename = argv[1];
	const char *outputFilenameSuffix = filenameSuffix(outputFilename);

	const int kOutputWidth  = 800;
	const int kOutputHeight = 600;
#ifdef _DEBUG
	const int kSamplesPerPixel = 10;
#else
	const int kSamplesPerPixel = 100;
#endif

	std::default_random_engine randomGen;
	randomGen.seed((unsigned long)std::chrono::high_resolution_clock::now().time_since_epoch().count()); // TODO(cort): seed correctness
	std::uniform_real_distribution<float> randomPixelOffsetX(-1.0f/(float)kOutputWidth,  1.0f/(float)kOutputWidth);
	std::uniform_real_distribution<float> randomPixelOffsetY(-1.0f/(float)kOutputHeight, 1.0f/(float)kOutputHeight);

	LambertianMaterial yellowLambert(float3(1,1,0.0));
	LambertianMaterial greenLambert(float3(0.3, 0.8, 0.3));
	MetalMaterial copperMetal(float3(0.8549f, 0.5412f, 0.4039f), 0.4f);
	MetalMaterial silverMetal(float3(0.9,0.9,0.9), 0.05f);
	HitteeList hittees( std::vector<Hittee*>{
		new Sphere( float3(0.0f, -100, 00.0f), 100.0f, &greenLambert ),
		new Sphere( float3(0.0f, 0.5f, 0.0f), 0.5f, &yellowLambert ),
		new Sphere( float3(-1.0f, 0.5f, 0.0f), 0.5f, &copperMetal ),
		new Sphere( float3( 1.0f, 0.5f, 0.0f), 0.5f, &silverMetal ),
	});

	const float aspectRatio = (float)kOutputWidth / (float)kOutputHeight;
	const float3 camPos    = float3( 0, 1, 5);
	const float3 camTarget = float3(0,0,0);
	const float3 camUp     = float3( 0, 1, 0);
	Camera camera(camPos, camTarget, camUp, 45.0f, aspectRatio);

	float *outputPixels = new float[kOutputWidth * kOutputHeight * 4];

	LARGE_INTEGER startTime, endTime, timerFreq;
	QueryPerformanceFrequency(&timerFreq);
	QueryPerformanceCounter(&startTime);
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
	QueryPerformanceCounter(&endTime);

	int imageWriteSuccess = 0;
	if (strncmp(outputFilenameSuffix, "hdr", 3) == 0)
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
		if (strncmp(outputFilenameSuffix, "bmp", 3) == 0)
			imageWriteSuccess = stbi_write_bmp(outputFilename, kOutputWidth, kOutputHeight, 4, normalizedPixels);
		if (strncmp(outputFilenameSuffix, "png", 3) == 0)
			imageWriteSuccess = stbi_write_png(outputFilename, kOutputWidth, kOutputHeight, 4, normalizedPixels, kOutputWidth*sizeof(uint32_t));
		if (strncmp(outputFilenameSuffix, "tga", 3) == 0)
			imageWriteSuccess = stbi_write_tga(outputFilename, kOutputWidth, kOutputHeight, 4, normalizedPixels);
		delete [] normalizedPixels;
	}
	assert(imageWriteSuccess);
	delete [] outputPixels;

	double elapsed = double(endTime.QuadPart-startTime.QuadPart) / double(timerFreq.QuadPart);
	printf("Rendered %s [%dx%d %ds/p] in %.3f seconds\n", outputFilename, kOutputWidth, kOutputHeight, kSamplesPerPixel, elapsed);
	return 0;
}
