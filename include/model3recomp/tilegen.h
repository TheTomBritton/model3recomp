/* model3recomp -- Model 3 tilemap generator.
 *
 * Four 8x8 tile layers over the 496x384 display: the service menu, the HUD,
 * and the flat parts of attract mode. Called once per field, before the
 * framebuffer is presented.
 */
#ifndef MODEL3RECOMP_TILEGEN_H
#define MODEL3RECOMP_TILEGEN_H

#ifdef __cplusplus
extern "C" {
#endif

/* One priority pass: the layers the guest has put behind the 3D, or the
 * ones it has put in front. */
void tilegen_render_pass(int above);

/* The same, for screen rows ry0..ry1 only: rows are independent, so threads
 * can share a pass a band of rows each. */
void tilegen_render_rows(int above, int ry0, int ry1);

/* For a host that composes a field on other threads while the guest runs the
 * next one: copy VRAM and the registers as they stand into copy k (0 or 1),
 * and draw from copy k (or from the live ones, -1). */
void tilegen_snapshot(int k);
void tilegen_set_source(int k);

/* For bring-up: isolate a layer when working out which one holds what. */
void tilegen_enable_layer(int layer, int on);

#ifdef __cplusplus
}
#endif
#endif
