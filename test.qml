import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang
import Qt.labs.platform

ApplicationWindow {
  id: root
  title: "Ising Simulation"
  width: 256
  height: 256
  visible: true
  // Text{
  //     text: obs.txt
  // }
  // Text{
  //     text: obs.txt2
  // }

  JuliaCanvas{
    id: canvas
    width: 10
    height: 10
    paintFunction: showlatest
  }
}