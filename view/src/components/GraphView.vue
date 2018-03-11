<template>
<div>
  <select v-model="treeIndex">
    <option
    v-for="(tree, i) in treeData.length"
    :key="tree">
    {{ i }}
    </option>
  </select>
  <div
  class="cyto-container"
  ref="cyto"></div>
</div>
</template>
<script>
import treeData from '../assets/node_edge.json'
import dagre from 'cytoscape-dagre'
var cytosape = require('cytoscape')
cytosape.use(dagre)

export default {
  data () {
    return {
      treeIndex: 0,
      treeData: treeData,
      cyto: null
    }
  },
  watch: {
    treeIndex (newVal) {
      this.treeIndex = newVal
      const newData = this.treeData[this.treeIndex]

      this.cyto.remove('*')
      for (let t of ['nodes', 'edges']) {
        this.cyto.add(
          newData[t].map((v) => {
            v['group'] = t
            return v
          }))
      }
      const layout = this.cyto.makeLayout({
        name: 'dagre',
        animate: false
      })
      layout.run()
    }
  },
  mounted () {
    const dom = this.$refs.cyto
    console.log(dom)
    this.cyto = cytosape({
      container: dom,
      boxSelectionEnabled: false,
      autounselectify: true,
      layout: {
        name: 'dagre',
        animate: true
      },

      style: [
        {
          selector: 'node',
          style: {
            'content': 'data(gain)',
            'text-opacity': 0.5,
            'text-valign': 'center',
            'text-halign': 'right',
            'background-color': '#11479e'
          }
        },

        {
          selector: 'edge',
          style: {
            'curve-style': 'bezier',
            'width': 4,
            'target-arrow-shape': 'triangle',
            'line-color': '#9dbaea',
            'target-arrow-color': '#9dbaea'
          }
        }
      ],
      elements: this.treeData[this.treeIndex]
    })
  }
}
</script>

<style>
.cyto-container {
  width: 100%;
  height: 60vh;
}
</style>
