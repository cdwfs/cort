#include "cds_float3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _MSC_VER
#	include <windows.h>
#endif

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#if !defined(M_PI)
#	define M_PI 3.14159265358979323846
#endif

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

struct HitRecord
{
	float t;
	float3 pos;
	float3 normal;
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

class Sphere : public Hittee
{
public:
	Sphere() {}
	explicit Sphere(float3 center, float radius) : center(center), radius(radius) {}
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
				return true;
			}
			float t1 = (-b + temp) / a;
			if (tMin <= t1 && t1 <= tMax)
			{
				outRecord->t = t1;
				outRecord->pos = ray.eval(t1);
				outRecord->normal = (outRecord->pos - center) / radius;
				return true;
			}
		}
		return false;
	}

	float3 center;
	float radius;
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

float3 CDSF3_VECTORCALL hitColor(const HitRecord &hit)
{
	return (hit.normal+float3(1,1,1)) * 0.5f;
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

	HitteeList hittees( std::vector<Hittee*>{
		new Sphere( float3(0.0f, 0.5f, 0.0f), 0.5f ),
		new Sphere( float3(0.0f, -100, 00.0f), 100.0f ),
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
			Ray ray = camera.ray(u,v);
			float3 color(0,0,0);

			HitRecord hit;
			if (hittees.hit(ray, 0, FLT_MAX, &hit))
			{
				color = hitColor(hit);
			}
			else
			{
				// background color
				float3 unitDir = normalize(ray.dir);
				float t = 0.5f * (unitDir.y() + 1.0f);
				color = lerp( float3(1,1,1), float3(0.5, 0.7, 1.0), t);
			}

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
			normalizedPixels[i] =
				( uint8_t(255.0f*color.r()) <<  0 ) |
				( uint8_t(255.0f*color.g()) <<  8 ) |
				( uint8_t(255.0f*color.b()) << 16 ) |
				( 0xFF                      << 24 );
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
	printf("Rendered %s [%dx%d] in %.3f seconds\n", outputFilename, kOutputWidth, kOutputHeight, elapsed);
	return 0;
}
