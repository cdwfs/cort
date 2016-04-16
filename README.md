Cort's Obligatory Ray Tracer
============================

![image](https://raw.githubusercontent.com/cdwfs/cort/master/ref.jpg)

I hadn't written a ray tracer in nearly twenty years, so when I found [Peter Shirley](https://www.cs.utah.edu/~shirley/)'s [Ray Tracing in One Weekend](http://www.amazon.com/Ray-Tracing-Weekend-Peter-Shirley-ebook/dp/B01B5AODD8?ie=UTF8&keywords=ray%20tracing%20in%20one%20weekend&qid=1458891129&ref_=sr_1_1&sr=8-1) ebook (and its [sequels](http://in1weekend.com)), I figured it was time to rectify that. I don't usually have any free time on weekends, so I wrote mine between meetings at GDC instead.

The result is nothing remotely fancy or groundbreaking, but now I've caught the ray-tracing bug again.

At the moment I'm only about halfway through the second book. The following features are implemented:
 * Spheres out the wazoo
 * Lambertian, dielectric, and metallic materials
 * Super-sampled anti-aliasing
 * Depth of field
 * Load-balanced multi-threaded rendering
 * BVH-accelerated spatial queries
 
Thankses
========
 * [Peter Shirley](https://www.cs.utah.edu/~shirley/) for the [In One Weekend](http://in1weekend.com) series
 * [Sean Barrett](http://nothings.org) for [stb_image_write.h](https://github.com/nothings/stb)
 * [Richard Mitton](http://www.codersnotes.com/) for [How To Write A Maths Library In 2016](http://www.codersnotes.com/notes/maths-lib-2016/)
