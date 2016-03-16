#include "cds_float3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef _MSC_VER
#	include <windows.h>
#endif

#include <assert.h>
#include <math.h>
#include <stdio.h>

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

bool intersectRaySphere(float3 center, float radius, Ray r, float *hitT)
{
	float3 oc = r.origin - center;
	float a = dot(r.dir, r.dir);
	float b = 2.0f * dot(oc, r.dir);
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b - 4*a*c;
	if (discriminant < 0)
		return false;
	*hitT = (-b - sqrtf(discriminant)) / 2.0f*a;
	return true;
}

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

const char *filenameSuffix(const char *filename)
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

	const float aspectRatio = (float)kOutputWidth / (float)kOutputHeight;
	const float3 camPos    = float3( 0, 0, 0);
	const float3 camTarget = sphere.center;
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
			Ray r = camera.ray(u,v);
			float3 color(0,0,0);

			float3 sphereCenter = float3(0, 0.5f, -5);
			float sphereRadius = 0.3f;
			float sphereHitT = 0;
			if (intersectRaySphere(sphereCenter, sphereRadius, r, &sphereHitT))
			{
				float3 normal = normalize(r.eval(sphereHitT) - sphereCenter);
				color = (normal+float3(1,1,1)) * 0.5f;
			}
			else
			{
				float3 unitDir = normalize(r.dir);
				float t = 0.5f * (unitDir.y() + 1.0f);
				color = lerp( float3(1,1,1), float3(0,0,0), t);
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
