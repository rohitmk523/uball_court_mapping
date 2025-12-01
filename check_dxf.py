import ezdxf

dxf_file = "court_2.dxf"
doc = ezdxf.readfile(dxf_file)
msp = doc.modelspace()

# Count entity types
entity_types = {}
for entity in msp:
    etype = entity.dxftype()
    entity_types[etype] = entity_types.get(etype, 0) + 1

print("DXF Entity Types Found:")
print("="*50)
for etype, count in sorted(entity_types.items()):
    print(f"{etype:20s}: {count:3d}")

print("\n" + "="*50)
print("\nDetailed Entity Information:")
print("="*50)

# Show details for each entity type
for entity in msp:
    etype = entity.dxftype()
    if etype == 'ARC':
        print(f"\nARC:")
        print(f"  Center: ({entity.dxf.center.x:.2f}, {entity.dxf.center.y:.2f})")
        print(f"  Radius: {entity.dxf.radius:.2f}")
        print(f"  Start angle: {entity.dxf.start_angle:.2f}°")
        print(f"  End angle: {entity.dxf.end_angle:.2f}°")

