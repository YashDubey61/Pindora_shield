import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

function MoleculeModel() {
  return (
    <>
      {/* Atom A */}
      <mesh position={[-1.2, 0, 0]}>
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshStandardMaterial color="#ff4d4d" />
      </mesh>

      {/* Atom B */}
      <mesh position={[1.2, 0, 0]}>
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshStandardMaterial color="#4dff4d" />
      </mesh>

      {/* Bond */}
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.1, 0.1, 2.4, 32]} />
        <meshStandardMaterial color="#cccccc" />
      </mesh>
    </>
  );
}

export default function MoleculeViewer() {
  return (
    <div style={{ width: "100vw", height: "100vh", background: "#0f172a" }}>
      <Canvas camera={{ position: [0, 0, 6] }}>
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 5, 5]} intensity={1} />
        <MoleculeModel />
        <OrbitControls />
      </Canvas>
    </div>
  );
}