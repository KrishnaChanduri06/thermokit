"""Illustrative desktop-CPU style stack. Values are typical, not from a datasheet."""

from thermokit import Contact, Fixed, Slab, Spreader, solve

mm = 1e-3
die = 12 * mm * 12 * mm
lid = 35 * mm * 35 * mm
base = 60 * mm * 60 * mm

layers = [
    Fixed("junction-case", 0.10),
    Contact("TIM2 paste", 0.10e-4, lid),          # R'' = 0.10 K·cm²/W
    Spreader("Cu base 6 mm", lid, base, 6 * mm, 390),
    Fixed("fins + air", 0.18),
]

result = solve(layers, power=125.0, ambient=35.0)
print(result.table())

print("\nIf the base were aluminium instead:")
layers[2] = Spreader("Al base 6 mm", lid, base, 6 * mm, 200)
print(solve(layers, power=125.0, ambient=35.0).table())
