import Formal.Shells
import Formal.SelectionRules

/-!
Reference outputs of the executable Lean models. Rust tests compare `geometry::cube`,
`waves::degrees` and `translation_plan::harmonics` against them. `lake exe golden`
rewrites `golden/`; `lake exe golden --check` fails when a file is stale.
-/

open Treams

def cube : String := String.join <|
  ([1, 2, 3] : List ℕ).flatMap fun dim => ([0, 1, 2, 3, 4] : List ℤ).flatMap fun n =>
    [true, false].map fun edge =>
      s!"{dim} {n} {edge}:" ++
        String.join ((Shells.cube dim n edge).map fun x => " " ++ ",".intercalate (x.map toString))
        ++ "\n"

def degrees : String := String.join <|
  (irange 1 5).flatMap fun l => (irange 1 5).flatMap fun lam =>
    (irange (-l) l).flatMap fun m => (irange (-lam) lam).flatMap fun mu =>
      ([0, 1] : List ℤ).map fun c =>
        s!"{l} {m} {lam} {mu} {c}:" ++
          String.join ((SelectionRules.termDegrees l m lam mu c).map fun p => s!" {p}") ++ "\n"

def harmonics : String := String.join <|
  (List.range 9).map fun order =>
    s!"{order}:" ++ String.join ((Harmonics.table order).map fun (p, m) => s!" {p},{m}") ++ "\n"

def main (args : List String) : IO UInt32 := do
  let files := [("golden/cube.txt", cube), ("golden/degrees.txt", degrees),
    ("golden/harmonics.txt", harmonics)]
  if args == ["--check"] then
    let mut stale := false
    for (path, content) in files do
      let current ← try IO.FS.readFile path catch _ => pure ""
      if current != content then
        IO.eprintln s!"{path} is stale; run `lake exe golden` in formal/"
        stale := true
    return if stale then 1 else 0
  IO.FS.createDirAll "golden"
  for (path, content) in files do
    IO.FS.writeFile path content
  return 0
